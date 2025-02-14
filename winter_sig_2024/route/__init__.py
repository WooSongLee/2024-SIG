from fastapi import APIRouter
from .auth import router as auth_router
from .diary import router as diary_router
from .audio import router as voice_router
from .image import router as image_router

api_router = APIRouter()

api_router.include_router(auth_router, prefix="/auth", tags=["Auth"])
api_router.include_router(diary_router, prefix="/diary", tags=["Diary"])
api_router.include_router(voice_router, prefix="/voice", tags=["Voice"])
api_router.include_router(image_router, prefix="/image", tags=["Image"])
