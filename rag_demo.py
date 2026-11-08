
from pathlib import Path

import chromadb
from ollama import chat, embed

# Project paths
project_folder = Path(__file__).resolve().parent
document_path = project_folder / "document.txt"
database_path = project_folder / "chroma_db"

# Load document
if not document_path.exists():
    raise FileNotFoundError(f"Document not found: {document_path}")

document = document_path.read_text(encoding="utf-8")

if not document.strip():
    raise ValueError("document.txt is empty.")

# Split document into overlapping chunks
def chunk_text(text, chunk_size=500, overlap=80):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


chunks = chunk_text(document)
print(f"Loaded {len(chunks)} document chunks.")

# Create a persistent local vector database
client = chromadb.PersistentClient(path=str(database_path))

# Rebuild the collection whenever this script runs
try:
    client.delete_collection(name="documents")
except Exception:
    pass

collection = client.create_collection(name="documents")

# Convert document chunks into embeddings
embedding_response = embed(
    model="all-minilm",
    input=chunks
)

collection.add(
    ids=[f"chunk-{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embedding_response.embeddings,
    metadatas=[
        {"source": document_path.name, "chunk": i}
        for i in range(len(chunks))
    ],
)

print("Documents indexed successfully!")

# Ask questions
while True:
    question = input("\nAsk a question (or type 'exit'): ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question:
        continue

    try:
        # Convert the question into an embedding
        question_embedding = embed(
            model="all-minilm",
            input=question
        ).embeddings[0]

        # Retrieve the three most relevant chunks
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=min(3, len(chunks))
        )

        relevant_chunks = results["documents"][0]
        context = "\n\n".join(relevant_chunks)

        # Give the retrieved context to Qwen
        response = chat(
            model="qwen2.5:1.5b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer using only the supplied context. "
                        "If the answer is not present in the context, "
                        "say you don't know based on the provided documents. "
                        "Do not invent facts."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"CONTEXT:\n{context}\n\n"
                        f"QUESTION:\n{question}"
                    ),
                },
            ],
        )

        print("\nAI Answer:")
        print(response.message.content)

    except Exception as error:
        print(f"Error: {error}")
