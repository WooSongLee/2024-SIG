from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..DB.db import get_db
from ..services.auth_service import login_user

router = APIRouter()

# 로그인 요청 데이터 구조
class LoginRequest(BaseModel):
    id: str
    password: str

@router.post("/login")
def login(data: LoginRequest):
    login_user(data.id, data.password, get_db())