from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# 이미지 요청 데이터 모델
class ImageRequest(BaseModel):
    date: str
    title: str
    content: str

@router.post("/generate-image")
def generate_image(data: ImageRequest):
    return {
        "image_url1": f"http://localhost:8000/static/images/{data.date}-1.jpg",
        "image_url2": f"http://localhost:8000/static/images/{data.date}-2.jpg",
        "image_url3": f"http://localhost:8000/static/images/{data.date}-3.jpg"
    }
