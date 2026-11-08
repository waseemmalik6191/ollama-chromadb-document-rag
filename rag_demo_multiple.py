
from pathlib import Path

import chromadb
from ollama import chat, embed

# Project folders
project_folder = Path(__file__).resolve().parent
documents_folder = project_folder / "documents"
database_path = project_folder / "chroma_db"

if not documents_folder.exists():
    raise FileNotFoundError(
        f"Documents folder not found: {documents_folder}"
    )

# Find all text files
document_paths = sorted(documents_folder.rglob("*.txt"))

if not document_paths:
    raise ValueError(
        f"No .txt files found in {documents_folder}"
    )

# Split text into overlapping chunks
def chunk_text(text, chunk_size=500, overlap=80):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# Read and process every document
all_chunks = []
all_ids = []
all_metadatas = []

for document_path in document_paths:
    text = document_path.read_text(encoding="utf-8")

    if not text.strip():
        print(f"Skipping empty file: {document_path.name}")
        continue

    chunks = chunk_text(text)

    for index, chunk in enumerate(chunks):
        relative_path = document_path.relative_to(
            documents_folder
        ).as_posix()

        chunk_id = f"{relative_path}-chunk-{index}"

        all_chunks.append(chunk)
        all_ids.append(chunk_id)
        all_metadatas.append({
            "source": relative_path,
            "chunk": index
        })

    print(f"Loaded: {document_path.name} ({len(chunks)} chunks)")

if not all_chunks:
    raise ValueError("No usable text chunks found.")

print(f"\nTotal files: {len(document_paths)}")
print(f"Total chunks: {len(all_chunks)}")

# Create the local vector database
client = chromadb.PersistentClient(
    path=str(database_path)
)

# Rebuild the collection each time this script runs
try:
    client.delete_collection(name="documents")
except Exception:
    pass

collection = client.create_collection(name="documents")

# Generate embeddings for all chunks in batches
batch_size = 100

for start in range(0, len(all_chunks), batch_size):
    end = start + batch_size

    batch_chunks = all_chunks[start:end]
    batch_embeddings = embed(
        model="all-minilm",
        input=batch_chunks
    ).embeddings

    collection.add(
        ids=all_ids[start:end],
        documents=batch_chunks,
        embeddings=batch_embeddings,
        metadatas=all_metadatas[start:end]
    )

print("\nAll documents indexed successfully!")

# Ask questions across all files
while True:
    question = input(
        "\nAsk a question (or type 'exit'): "
    ).strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question:
        continue

    try:
        question_embedding = embed(
            model="all-minilm",
            input=question
        ).embeddings[0]

        # Retrieve the three nearest chunks
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=min(3, collection.count())
        )

        relevant_chunks = results["documents"][0]
        metadatas = results["metadatas"][0]

        context_parts = []

        for chunk, metadata in zip(
            relevant_chunks, metadatas
        ):
            context_parts.append(
                f"Source: {metadata['source']}\n{chunk}"
            )

        context = "\n\n".join(context_parts)

        response = chat(
            model="qwen2.5:1.5b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer using only the supplied context. "
                        "If the answer is not present, say you "
                        "don't know based on the provided documents. "
                        "Do not invent facts. Mention the source "
                        "filename when useful."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"CONTEXT:\n{context}\n\n"
                        f"QUESTION:\n{question}"
                    )
                }
            ]
        )

        print("\nAI Answer:")
        print(response.message.content)

        print("\nRetrieved sources:")
        for metadata in metadatas:
            print(f"- {metadata['source']}")

    except Exception as error:
        print(f"Error: {error}")
