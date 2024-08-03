import os
from fastapi import UploadFile
import ffmpeg
from pathlib import Path

from constans import UPLOAD_DIRECTORY


class FileManager:
    def _convert_webm_to_wav(self, webm_file_path: str):
        if Path(webm_file_path).suffix == '.wav':
            return webm_file_path

        new_path = Path(webm_file_path).with_suffix('.wav')
        ffmpeg.input(webm_file_path).output(str(new_path), y='-y').run(overwrite_output=True)
        return str(new_path)

    async def write_file_locally(self, file: UploadFile) -> str:
        os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)

        with open(file_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)

        return self._convert_webm_to_wav(file_path)

    def rename_file(self, original_filename: str, new_filename: str) -> None:
        try:
            original_path = Path(original_filename)
            new_path = original_path.with_name(Path(new_filename).name)
            original_path.rename(new_path)
        except FileNotFoundError:
            pass

    def remove_file(self, file_path: str):
        try:
            Path(file_path).unlink()
        except FileNotFoundError:
            pass

    def remove_uploaded_files_directory_files(self):
        directory = Path(UPLOAD_DIRECTORY)

        for file in directory.iterdir():
            if file.is_file() or file.is_symlink():
                file.unlink()
            elif file.is_dir():
                file.rmdir()


file_manager = FileManager()
