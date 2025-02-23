import io
import speech_recognition as sr
from fastapi import UploadFile
from pydub import AudioSegment

def extract_contents(audio_file: UploadFile) -> str:
    recognizer = sr.Recognizer()

    audio_bytes = audio_file.file.read()

    audio_data = convert_to_wav(audio_bytes)

    with sr.AudioFile(audio_data) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio, language="ko-KR")
        return text
    except sr.UnknownValueError:
        return "음성을 인식할 수 없습니다."
    except sr.RequestError:
        return "구글 음성 인식 서비스에 요청할 수 없습니다."

def convert_to_wav(audio_bytes: bytes) -> io.BytesIO:
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))

    wav_audio = io.BytesIO()
    audio.export(wav_audio, format="wav")
    wav_audio.seek(0)

    return wav_audio
