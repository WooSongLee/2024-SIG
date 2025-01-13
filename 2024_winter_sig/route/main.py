from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from ..DB.db import SessionLocal, Base, engine
from ..services.login import register_user, login_user

app = FastAPI()

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/register/")
def register(user_id: str, user_pwd: str, user_name: str, db: Session = Depends(get_db)):
    return register_user(user_id, user_pwd, user_name, db)

@app.post("/login/")
def login(user_id: str, user_pwd: str, db: Session = Depends(get_db)):
    return login_user(user_id, user_pwd, db)
