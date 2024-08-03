import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from pydub import AudioSegment

# Assuming the SpeechToText class is defined in a module called transcribe
from src.transcribe import SpeechToText, SECOND_FILE_PATH, THIRD_FILE_PATH, FIRST_FILE_PATH, COMBINED_FILE_PATH


@pytest.fixture
def mock_speech_to_text():
    with patch.object(SpeechToText, '__init__', lambda x: None):
        instance = SpeechToText()
        instance.init_whisper_model_and_processor = MagicMock()
        return instance


@patch("transcribe.Path.is_file")
@patch("transcribe.AudioSegment.from_file")
@patch("transcribe.AudioSegment.export")
def test_get_file_path_for_transcribe_no_first_file(mock_export, mock_from_file, mock_is_file, speech_to_text):
    # Simulate SECOND_FILE_PATH exists but FIRST_FILE_PATH does not
    mock_is_file.side_effect = lambda p: p == Path(SECOND_FILE_PATH)

    # Mock the AudioSegment instances
    mock_sound1 = MagicMock()
    mock_sound2 = MagicMock()
    mock_from_file.side_effect = [mock_sound1, mock_sound2]

    result = speech_to_text.get_file_path_for_transcribe()

    # Ensure the combined file path is returned
    assert result == COMBINED_FILE_PATH

    # Ensure AudioSegment.from_file was called twice
    mock_from_file.assert_any_call(SECOND_FILE_PATH, format="wav")
    mock_from_file.assert_any_call(THIRD_FILE_PATH, format="wav")

    # Ensure the combined sound was exported
    mock_sound1.__add__.assert_called_once_with(mock_sound2)
    mock_export.assert_called_once_with(COMBINED_FILE_PATH, format="wav")


@patch("transcribe.Path.is_file")
def test_get_file_path_for_transcribe_no_second_file(mock_is_file, speech_to_text):
    # Simulate SECOND_FILE_PATH does not exist
    mock_is_file.return_value = False

    result = speech_to_text.get_file_path_for_transcribe()

    # Ensure the third file path is returned
    assert result == THIRD_FILE_PATH


@patch("transcribe.Path.is_file")
@patch("transcribe.AudioSegment.from_file")
@patch("transcribe.AudioSegment.export")
def test_get_file_path_for_transcribe_all_files_exist(mock_export, mock_from_file, mock_is_file, speech_to_text):
    # Simulate all files exist
    mock_is_file.return_value = True

    # Mock the AudioSegment instances
    mock_sound1 = MagicMock()
    mock_sound2 = MagicMock()
    mock_sound3 = MagicMock()
    mock_from_file.side_effect = [mock_sound1, mock_sound2, mock_sound3]

    result = speech_to_text.get_file_path_for_transcribe()

    # Ensure the combined file path is returned
    assert result == COMBINED_FILE_PATH

    # Ensure AudioSegment.from_file was called three times
    mock_from_file.assert_any_call(FIRST_FILE_PATH, format="wav")
    mock_from_file.assert_any_call(SECOND_FILE_PATH, format="wav")
    mock_from_file.assert_any_call(THIRD_FILE_PATH, format="wav")

    # Ensure the combined sound was exported
    mock_sound1.__add__.assert_called_once_with(mock_sound2)
    mock_sound1.__add__().assert_called_once_with(mock_sound3)
    mock_export.assert_called_once_with(COMBINED_FILE_PATH, format="wav")
