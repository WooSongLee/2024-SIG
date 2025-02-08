from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# 로그인 요청 데이터 구조
class LoginRequest(BaseModel):
    id: str
    password: str

@router.post("/login")
def login(data: LoginRequest):
    #로그인 확인 및 토큰 발급
    if data.id == "test" and data.password == "1234":
        return {"success": True, "token": "example_token"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")
