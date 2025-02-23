from fastapi import APIRouter, UploadFile, File, Form
from winter_sig_2024.services.voice_service import extract_contents
from winter_sig_2024.NLP_processing.main import generateText

router = APIRouter()

@router.post("/upload-audio")
def upload_audio(audio_file: UploadFile = File(...), date: str = Form(...), tone: str = Form(...)):
    #1. 음성 파일 받아서 문자열 추출
    extractedContents = extract_contents(audio_file)
    #2. 문체변환및 문법교정
    content = generateText(extractedContents, tone)

    return {
        "status": True,
        "date": date,
        "tone": tone,
        "content": content,
        "prevContent" : extractedContents
    }
