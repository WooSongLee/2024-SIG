from sqlalchemy import Column, Integer, String
from .db import Base

class User(Base):
    __tablename__ = "User"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(45), unique=True, nullable=False)
    user_pwd = Column(String(45), nullable=False)
    user_name = Column(String(45), nullable=False)

