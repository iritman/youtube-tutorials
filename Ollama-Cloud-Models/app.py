import ollama
import os

os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

client = ollama.Client(host="127.0.0.1:11434")

model = "deepseek-v3.1:671b-cloud"

while True:
    user_input = input("User: ")

    if (user_input == "exit"):
        break

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "user", 
                "content": user_input
            },
        ],
    )

    print(response.message.content)