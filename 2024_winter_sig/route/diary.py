from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from ..services.diary_service import getDiary, getMainDiary, getSelectedDiary

router = APIRouter()

class SavingData(BaseModel):
    date: str
    image_url : str
    title : str
    contents : str

@router.get("/diary")
def get_diary():
    return getDiary()

@router.get("/MainDiary")
def get_main_diary():
    return getMainDiary()

@router.get("/diary/${selectedDate}")
def get_selected_diary(selectedDate : str):
    return getSelectedDiary(selectedDate)

@router.get("/saving")
def saving_diary(savingdata: SavingData):
    #return status