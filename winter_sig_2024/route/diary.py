from fastapi import APIRouter, Depends
from winter_sig_2024.schemas import SavingData
from winter_sig_2024.services.diary_service import getDiary, getMainDiary, getSelectedDiary, savingDiary,getAllDiaries
from winter_sig_2024.DB.db import get_db
from sqlalchemy.orm import Session

router = APIRouter()


@router.get("/diary")
def get_diary(db: Session = Depends(get_db)):
    return getDiary(db)


@router.get("/MainDiary")
def get_main_diary(db: Session = Depends(get_db)):
    return getMainDiary(db)


@router.get("/diary/${selectedDate}")
def get_selected_diary(selectedDate: str, db: Session = Depends(get_db)):
    return getSelectedDiary(db, selectedDate)


@router.get("/saving")
def saving_diary(savingdata: SavingData, db: Session = Depends(get_db)):
    return savingDiary(db, savingdata)

@router.get("/allDiary")
def get_all_diary(db: Session = Depends(get_db)):
    return getAllDiaries(db)