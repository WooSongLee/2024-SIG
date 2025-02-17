import datetime
from sqlalchemy.orm import Session
from winter_sig_2024.DB.models import Diary
from winter_sig_2024.schemas import SavingData

# 이번달 일기 날짜, 제목, 이미지 return
def getDiary(db: Session):
    now = datetime.now()
    year, month = now.year, now.month

    diaries = db.query(Diary).filter(Diary.diary_date.year == year, Diary.diary_date.month == month).all()

    return [
        {
            "date": diary.diary_date.strftime("%Y-%m-%d"),
            "title": diary.diary_title,
            "image": diary.diary_image.hex() if diary.diary_image else None
        }
        for diary in diaries
    ]


# 메인에 들어갈 일기달력(이번달 작성한 일기가 존재하는 날짜 return)
def getMainDiary(db: Session):
    now = datetime.now()
    year, month = now.year, now.month

    diaries = db.query(Diary.diary_date).filter(Diary.diary_date.year == year, Diary.diary_date.month == month).all()

    diary_dates = [diary.diary_date.strftime("%Y-%m-%d") for diary in diaries]

    return {"dates": diary_dates}


# 선택한 날짜의 일기 제목, 내용, image return
def getSelectedDiary(db: Session, selectedDate: str):
    diary = db.query(Diary).filter(Diary.diary_date == selectedDate).first()

    return {
        "date": diary.diary_date.strftime("%Y-%m-%d"),
        "title": diary.diary_title,
        "content": diary.diary_content,
        "image": diary.diary_image.hex() if diary.diary_image else None
    }


def savingDiary(db: Session, data: SavingData):
    diary_date = data.date
    diary_image = data.image
    diary_title = data.title
    diary_contents = data.contents

    # 새로운 Diary 객체 생성
    new_diary = Diary(
        diary_date=diary_date,
        diary_image=diary_image,
        diary_title=diary_title,
        diary_content=diary_contents,
    )

    db.add(new_diary)
    db.commit()
    db.refresh(new_diary)

    return {"status": "success"}
