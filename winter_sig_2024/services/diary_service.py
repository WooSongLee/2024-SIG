import datetime
from typing import List
from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select
from winter_sig_2024.DB.models import Diary
from winter_sig_2024.schemas import SavingData
import base64


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
    diaries = db.query(Diary.diary_date, Diary.diary_title).all()

    diary_list = [
        {"date": diary.diary_date.strftime("%Y-%m-%d"), "title": diary.diary_title or "일기"}
        for diary in diaries
    ]

    return {"diaries": diary_list}




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


def getAllDiaries(db: Session):
    result = db.execute(select(Diary.diary_date)).scalars().all()

    dates = sorted(set(result))

    return {"dates": [date.strftime("%Y-%m-%d") for date in dates]}


def getDiaryList(db: Session) -> List[dict]:
    try:
        diaries = db.query(Diary).all()

        diary_list = []
        for diary in diaries:
            image_url = None
            if diary.diary_image:
                image_url = "data:image/jpeg;base64," + base64.b64encode(diary.diary_image).decode('utf-8')

            diary_list.append({
                "id": diary.diary_id,
                "date": diary.diary_date.strftime("%Y-%m-%d %H:%M:%S"),
                "title": diary.diary_title,
                "img": image_url,  # 이미지 URL
            })

        return {"list": diary_list}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"일기 목록을 가져오는 데 실패했습니다: {str(e)}")