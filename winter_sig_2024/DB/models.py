import blob
import DateTime
from sqlalchemy import Column, Integer, String
from .db import Base

class User(Base):
    __tablename__ = "User"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(45), unique=True, nullable=False)
    user_pwd = Column(String(45), nullable=False)
    user_name = Column(String(45), nullable=False)

class Diary(Base):
    __tablename__ = "diary"
    diary_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, primary_key=True)
    diary_date = Column(DateTime, nullable=False)
    diary_content = Column(String(45), nullable=True)
    diary_voice = Column(String(45), nullable=True)
    diary_image = Column(blob, nullable=True)
    diary_title = Column(String(45), nullable=True)