import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import AIMessageChunk
from langchain.tools import tool

load_dotenv()
api_key = os.getenv("OLLAMA_API_KEY")
api_url = os.getenv("OLLAMA_API_URL")
api_model = os.getenv("OLLAMA_API_MODEL")


llm = init_chat_model(
    model=api_model,
    api_key=api_key,
    base_url=api_url,
)

@tool(
    'get_user_department_info',
    description="Look up department information for a user by ID.",
    return_direct=False
)
def get_user_department_info(user_id: str) -> str:
    match user_id:
        case "1":
            return "Sales"
        case "2":
            return "Marketing"
        case "3":
            return "HR"
        case _:
            return "No user department on file"



memory = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=[get_user_department_info],
    checkpointer=memory,
)

thread_config = {"configurable": {"thread_id": "test"}}

while True:
    user_message = input('User: ')
    if user_message == 'exit' or user_message == 'bye':
        print('Goodbye!\n')
        break
    elif user_message == 'clear':
        memory.delete_thread(thread_config["configurable"]["thread_id"])
        print("Memory cleared.\n")
        continue

    print('AI: ')

    for chunk, metadata in agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        thread_config,
        stream_mode="messages"
    ):
        if chunk.content and isinstance(chunk, AIMessageChunk):
            print(chunk.content, end="", flush=True)

    print("\n")
