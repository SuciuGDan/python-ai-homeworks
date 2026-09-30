import json

from pydantic import BaseModel, EmailStr, Field

class StudentValidator(BaseModel):
    nume: str = Field(min_length=2, max_length=255)
    varsta: int = Field(ge=0, le=120)
    email: EmailStr


#random pytest for validating LLM agent




