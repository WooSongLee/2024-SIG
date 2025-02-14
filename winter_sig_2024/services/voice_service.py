import speech_recognition as sr
from fastapi import UploadFile
import io

def extract_contents(audio_file: UploadFile) -> str:
    recognizer = sr.Recognizer()

    audio_bytes = audio_file.file.read()
    audio_data = io.BytesIO(audio_bytes)

    with sr.AudioFile(audio_data) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio, language="ko-KR")
        return text
    except sr.UnknownValueError:
        return "음성을 인식할 수 없습니다."
