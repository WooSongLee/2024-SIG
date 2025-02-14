from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# 이미지 요청 데이터 모델
class ImageRequest(BaseModel):
    date: str
    title: str
    content: str
#
# @router.post("/generate-image")
# def generate_image(data: ImageRequest):
#     #이미지3개 return
