from fastapi import APIRouter
from winter_sig_2024.schemas import SavingData
from winter_sig_2024.DB.db import get_db
from winter_sig_2024.services.diary_service import getDiary, getMainDiary, getSelectedDiary, savingDiary

router = APIRouter()


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
    return savingDiary(get_db(), savingdata)
