from fastapi import FastAPI
from route import api_router
import uvicorn

app = FastAPI()

# API 라우트 등록
app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# Base.metadata.create_all(bind=engine)
#
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()
#
# @app.post("/register/")
# def register(request: RegisterRequest, db: Session = Depends(get_db)):
#     return register_user(request.user_id, request.user_pwd, request.user_name, db)
#
# @app.post("/login/")
# def login(request: LoginRequest, db: Session = Depends(get_db)):
#     return login_user(request.user_id, request.user_pwd, db)
