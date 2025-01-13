from sqlalchemy.orm import Session
from ..DB.db import SessionLocal, Base
from passlib.hash import bcrypt
from fastapi import HTTPException
from sqlalchemy import Column, Integer, String

class User(Base):
    __tablename__ = "User"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(45), unique=True, nullable=False)
    user_pwd = Column(String(45), nullable=False)
    user_name = Column(String(45), nullable=False)


def register_user(user_id: str, user_pwd: str, user_name: str, db: Session):
    existing_user = db.query(User).filter(User.user_id == user_id).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User ID already registered")

    new_user = User(user_id=user_id, user_pwd=user_pwd, user_name=user_name)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User registered successfully"}


def login_user(user_id: str, user_pwd: str, db: Session):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")

    if user.user_pwd != user_pwd:
        raise HTTPException(status_code=400, detail="Invalid credentials")

    return {"message": "Login successful"}
