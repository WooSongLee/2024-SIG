from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from winter_sig_2024.DB.db import get_db
from winter_sig_2024.services.auth_service import login_user

router = APIRouter()

# 로그인 요청 데이터 구조
class LoginRequest(BaseModel):
    id: str
    password: str

@router.post("/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    return login_user(data.id, data.password, db)

