import os
import re
from fastapi import BackgroundTasks, FastAPI, UploadFile, Response, status
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from groq import RateLimitError
from src.gemini_helper import check_for_video_action
from src.file_manager import file_manager
from src.guide_tube import guide_tube
from src.transcribe import speech_to_text

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}
SUPPORTED_COMMANDS = os.environ["SUPPORTED_COMMANDS"]
app = FastAPI()
app.add_middleware(TrustedHostMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Include OPTIONS method
    allow_headers=["*"],  # Allow any headers
    expose_headers=["Content-Disposition"],  # Optionally expose additional headers
)


def find_commands(transcription_text: str) -> list[str]:
    found_commands = [command for command in SUPPORTED_COMMANDS if command in transcription_text.lower()]
    return found_commands


"""
Removes punctuation from the transcription.
"""


def sanitize_transcription(transcription):
    return re.sub(r'[^\w\s]', '', transcription)


def allowed_file(file_path: str) -> bool:
    filename = os.path.basename(file_path)
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.put("/transcribe", status_code=200)
async def create_upload_file(file: UploadFile, response: Response) -> str:
    file_path: str = await file_manager.write_file_locally(file)
    if not (allowed_file(file_path)):
        response.status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        return "File type is not supported"
    try:
        transcription_text: str = speech_to_text.transcribe_with_groq(file_path)
        print(f"{transcription_text.lower() = }")
    except RateLimitError:
        print("Rate limit exception raised, using local whisper")
        transcription_text: str = speech_to_text.transcribe_with_whisper(file_path)
    transcription_text = sanitize_transcription(transcription_text)
    found_commands: list[str] = find_commands(transcription_text)
    if not found_commands:
        gemini_recognized_action = check_for_video_action(transcription_text)
        if gemini_recognized_action != "false":
            found_commands.append(gemini_recognized_action.split()[0])

    return found_commands[0]


@app.put("/hi_guide_tube", status_code=200)
async def hi_guide_tube(file: UploadFile, background_tasks: BackgroundTasks) -> bool:
    file_path: str = await file_manager.write_file_locally(file)
    # guide_tube.update_guide_tube_file(file_path=file_path)
    background_tasks.add_task(guide_tube.update_guide_tube_file, file_path=file_path)

    if guide_tube.is_guide_tube():
        file_manager.remove_file(file_path)
        return True

    return False


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=4000)
