import os
from dotenv import load_dotenv
from collections import defaultdict

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import AIMessageChunk
from langchain.tools import tool
from langchain.agents.middleware import SummarizationMiddleware

load_dotenv()
api_key = os.getenv("OLLAMA_API_KEY")
api_url = os.getenv("OLLAMA_API_URL")
api_model = os.getenv("OLLAMA_API_MODEL")


llm = init_chat_model(
    model=api_model,
    api_key=api_key,
    base_url=api_url,
    stream_usage=True
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
    middleware=[SummarizationMiddleware(
        model=llm,
        trigger=("tokens", 4000),
        keep=("messages", 10)
    )],
)

thread_config = {"configurable": {"thread_id": "test"}}

token_usage_totals = defaultdict(int)
counted_usage_checkpoints = set()

def get_token_usage() -> dict[str, int]:
    input_tokens = token_usage_totals["input_tokens"]
    output_tokens = token_usage_totals["output_tokens"]
    total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens, 
        "output_tokens": output_tokens, 
        "total_tokens": total_tokens
    }

def reset_token_usage() -> None:
    token_usage_totals.clear()
    counted_usage_checkpoints.clear()

while True:
    user_message = input('User: ')
    if user_message == 'exit' or user_message == 'bye':
        print('Goodbye!\n')
        break
    elif user_message == 'clear':
        memory.delete_thread(thread_config["configurable"]["thread_id"])
        reset_token_usage()
        print("Memory cleared.\n")
        continue
    elif user_message == 'token':
        usage = get_token_usage()
        print(f"""
            Memory token usage:
            Input tokens: {usage["input_tokens"]}
            Output tokens: {usage["output_tokens"]}
            Total tokens: {usage["total_tokens"]}
        """)
        continue

    print('AI: ')

    for chunk, metadata in agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        thread_config,
        stream_mode="messages"
    ):
        checkpoint_ns = metadata["checkpoint_ns"]
        usage = getattr(chunk, "usage_metadata", None) or {}
        if usage and checkpoint_ns not in counted_usage_checkpoints:
            token_usage_totals["input_tokens"] += usage["input_tokens"]
            token_usage_totals["output_tokens"] += usage["output_tokens"]
            counted_usage_checkpoints.add(checkpoint_ns)

        if chunk.content and isinstance(chunk, AIMessageChunk):
            print(chunk.content, end="", flush=True)

    print("\n")



