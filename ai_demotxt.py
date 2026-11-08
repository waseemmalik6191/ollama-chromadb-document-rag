
from pathlib import Path
from ollama import chat

# 1. Locate the document beside this Python file
project_folder = Path(__file__).resolve().parent
document_path = project_folder / "document.txt"

# 2. Read the document from your local drive
if not document_path.exists():
    raise FileNotFoundError(
        f"Document not found: {document_path}"
    )

document = document_path.read_text(encoding="utf-8")

if not document.strip():
    raise ValueError("The document is empty.")

print(f"Loaded document: {document_path.name}")
print(f"Characters loaded: {len(document)}")

# 3. Ask questions repeatedly
while True:
    question = input("\nAsk a question (or type 'exit'): ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question:
        continue

    # 4. Send the document and question to local AI
    try:
        response = chat(
            model="qwen2.5:1.5b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer questions using only the supplied "
                        "document. If the answer is not present, "
                        "say you don't know based on this document. "
                        "Do not invent facts."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"DOCUMENT:\n{document}\n\n"
                        f"QUESTION:\n{question}"
                    ),
                },
            ],
        )

        print("\nAI Answer:")
        print(response.message.content)

    except Exception as error:
        print(f"Error communicating with Ollama: {error}")
