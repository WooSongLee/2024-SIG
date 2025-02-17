from fastapi import APIRouter
from winter_sig_2024.route.auth import router as auth_router
from winter_sig_2024.route.diary import router as diary_router
from winter_sig_2024.route.audio import router as voice_router
from winter_sig_2024.route.image import router as image_router

api_router = APIRouter()

api_router.include_router(auth_router, prefix="/auth", tags=["Auth"])
api_router.include_router(diary_router, prefix="/diary", tags=["Diary"])
api_router.include_router(voice_router, prefix="/voice", tags=["Voice"])
api_router.include_router(image_router, prefix="/image", tags=["Image"])
