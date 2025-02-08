from fastapi import APIRouter, UploadFile, File, Form

router = APIRouter()

@router.post("/upload-audio")
def upload_audio(audio_file: UploadFile = File(...), date: str = Form(...), tone: str = Form(...)):
    return {
        "status": "success",
        "date": date,
        "tone": tone,
        "content": "오늘은 뜻깊은 하루였다..."
    }
