from fastapi import APIRouter, UploadFile, File, Form

router = APIRouter()

@router.post("/upload-audio")
def upload_audio(audio_file: UploadFile = File(...), date: str = Form(...), tone: str = Form(...)):
    #음성 추출
    #문체변환및 문법교정
    #return status, date, tone, contents(생성된 일기)

