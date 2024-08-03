from pathlib import Path

from constans import GUIDE_TUBE_FILE_PATH, SECOND_FILE_PATH, FIRST_FILE_PATH, THIRD_FILE_PATH
from src.file_manager import file_manager
from transcribe import speech_to_text
from gemini_helper import check_for_video_action


class GuideTube:
    def is_guide_tube(self):
        if Path(GUIDE_TUBE_FILE_PATH).is_file():
            file_manager.remove_file(GUIDE_TUBE_FILE_PATH)
            return True

        return False

    def _is_text_has_hi_guide_tube(self, transcription_text: str) -> bool:
        lower_transcription = transcription_text.lower()
        return (("hi" in lower_transcription or "hey" in lower_transcription) 
                and "tube" in lower_transcription) or (
            ("guide" in lower_transcription or "guy" in lower_transcription)
            and "tube" in lower_transcription)

    def _update_guide_tube_file(self, transcription_text: str) -> bool:

        if self._is_text_has_hi_guide_tube(transcription_text):
            file_manager.remove_uploaded_files_directory_files()
            Path(GUIDE_TUBE_FILE_PATH).touch()
        else:
            file_manager.remove_file(GUIDE_TUBE_FILE_PATH)

    def update_guide_tube_file(self, file_path: str):
        file_manager.rename_file(SECOND_FILE_PATH, FIRST_FILE_PATH)
        file_manager.rename_file(THIRD_FILE_PATH, SECOND_FILE_PATH)
        file_manager.rename_file(file_path, THIRD_FILE_PATH)

        transcribe_file: str = speech_to_text.get_file_path_for_transcribe()

        transcription_text: str = speech_to_text.transcribe_with_groq(transcribe_file)
        print(transcription_text)
        if not self._is_text_has_hi_guide_tube(transcription_text):
            transcription_text = check_for_video_action(transcription_text)
            print(transcription_text)

        self._update_guide_tube_file(transcription_text)


guide_tube = GuideTube()
