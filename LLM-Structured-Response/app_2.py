import os
import instructor
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List

load_dotenv()

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class Person(BaseModel):
    name: str
    age: int
    addresses: List[Address]

api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("OPENROUTER_MODEL")
base_url = os.getenv("OPENROUTER_BASE_URL")

client = instructor.from_openai(
    OpenAI(
        api_key=api_key,
        base_url=base_url,
    ),
    mode=instructor.Mode.JSON,
)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Extract information exactly as provided. Do not make up missing information."
            },
            {"role": "user", 
            "content": """
                Extract: John Smith is 35 years old.
                He has homes at 123 Main St, Springfield, IL 62704 and
                456 Oak Ave, Chicago, IL 60601.
                """}
        ],
        response_model=Person,
    )
    print(response)
except Exception as e:
    print(f"Validation error: {e}")

