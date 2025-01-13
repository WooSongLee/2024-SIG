from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from starlette.middleware.cors import CORSMiddleware

from ..DB.db import SessionLocal, Base, engine
from ..services.login import register_user, login_user
from ..schemas import RegisterRequest, LoginRequest

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (배포 시 특정 도메인만 허용)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/register/")
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    return register_user(request.user_id, request.user_pwd, request.user_name, db)

@app.post("/login/")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    return login_user(request.user_id, request.user_pwd, db)
