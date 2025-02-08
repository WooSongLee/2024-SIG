from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

#수정필요
DATABASE_URL = "mysql+pymysql://root:1234@localhost/MyDiary"

engine = create_engine(DATABASE_URL, echo=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
