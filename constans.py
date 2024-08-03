import os
CURRENT_DIRECTORY = os.getcwd()
UPLOAD_DIRECTORY = f"{CURRENT_DIRECTORY}/uploaded_files"
FIRST_FILE_PATH = os.path.join(UPLOAD_DIRECTORY, "first.wav")
SECOND_FILE_PATH = os.path.join(UPLOAD_DIRECTORY, "second.wav")
THIRD_FILE_PATH = os.path.join(UPLOAD_DIRECTORY, "third.wav")
COMBINED_FILE_PATH = os.path.join(UPLOAD_DIRECTORY, "combined.wav")
GUIDE_TUBE_FILE_PATH = os.path.join(UPLOAD_DIRECTORY, "guide_tub")