import os
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()

class Person(BaseModel):
    name: str = Field(
        description="Name of the person",
        min_length=1,
        max_length=100
    )
    age: int = Field(
        description="Age of the person",
        gt=0,
        lt=120,
    )
    occupation: str = Field(description="Occupation of the person")

    @field_validator("name")
    def validate_name(cls, v):
        if ' ' not in v:
            raise ValueError("Name must include first and last name")
        return v

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
            "content": "Extract: John Smith is a 25 year old software engineer"}
        ],
        response_model=Person,
    )
    print(response)
except Exception as e:
    print(f"Validation error: {e}")

