
from ollama import chat

# 1. Text stored directly in our Python program
document = """
The Result Processor application processes result events.
It depends on the ObjectStore component to store and retrieve objects.
Before deployment, verify all runtime dependencies and run the tests.
The application should be tested after any significant changes.
"""

# 2. Ask a question
question = input("Ask a question about the document: ")

# 3. Send the document and question to our local AI model
response = chat(
    model="qwen2.5:1.5b",
    messages=[
        {
            "role": "system",
            "content": (
                "Answer using only the supplied document. "
                "If the answer is not present, say you don't know."
            ),
        },
        {
            "role": "user",
            "content": f"Document:\n{document}\n\nQuestion:\n{question}",
        },
    ],
)

# 4. Display the answer
print("\nAI Answer:")
print(response.message.content)
