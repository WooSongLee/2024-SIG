from sqlalchemy.orm import Session
from fastapi import HTTPException
from winter_sig_2024.DB.models import User


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
    if not user or user.user_pwd != user_pwd:
        return {"success": False}

    return {"success": True}
