from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

# 일기 응답 데이터 모델
class DiaryEntry(BaseModel):
    date: str
    title: str
    image_url: str

class DiaryResponse(BaseModel):
    month: str
    days: List[DiaryEntry]

@router.get("/diary")
def get_diary():
    return DiaryResponse(
        month="2024-01",
        days=[
            {"date": "2024-01-01", "title": "새해 첫 날", "image_url": "https://example.com/1.jpg"},
            {"date": "2024-01-02", "title": "새로운 시작", "image_url": "https://example.com/2.jpg"},
        ]
    )
