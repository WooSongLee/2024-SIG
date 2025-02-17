from pydantic import BaseModel

class SavingData(BaseModel):
    date: str
    image: str
    title: str
    contents: str
