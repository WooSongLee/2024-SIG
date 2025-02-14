from fastapi import APIRouter
from pydantic import BaseModel
from ..DB.db import get_db
from ..services.diary_service import getDiary, getMainDiary, getSelectedDiary, savingDiary

router = APIRouter()


class SavingData(BaseModel):
    date: str
    image: str
    title: str
    contents: str


@router.get("/diary")
def get_diary():
    return getDiary(get_db())


@router.get("/MainDiary")
def get_main_diary():
    return getMainDiary(get_db())


@router.get("/diary/${selectedDate}")
def get_selected_diary(selectedDate: str):
    return getSelectedDiary(get_db(), selectedDate)


@router.get("/saving")
def saving_diary(savingdata: SavingData):
    return savingDiary(savingdata)
