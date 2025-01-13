from pydantic import BaseModel

class RegisterRequest(BaseModel):
    user_id: str
    user_pwd: str
    user_name: str

class LoginRequest(BaseModel):
    user_id: str
    user_pwd: str
