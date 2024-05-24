import google.generativeai as genai
from run_whisper import SUPPORTED_COMMANDS
import os

GOOGLE_API_KEY= os.environ["GOOGLE_API_KEY"]
SYSTEM_INSTRUCTIONS= f"""
Given a sentence, return a single action word (for example: "start," "stop," "repeat") that the sentence implies.
The output is the action only. without any additional information.
The output should be one of the following actions:
1. {SUPPORTED_COMMANDS}
2. false for any other case where the output doesnt describe one of the words above.
*Examples:*

1. *Sentence:* Begin the process.
   *Result:* start

2. *Sentence:* Cease all operations.
   *Result:* stop

3. *Sentence:* Do it once more.
   *Result:* repeat

4. *Sentence:* Let's go again.
   *Result:* repeat

5. *Sentence:* Initiate the program.
   *Result:* start

6. *Sentence:* Put everything on hold.
   *Result:* stop

7. *Sentence:* Hi
   *Result:* false

8. *Sentence:* mkelvnldr
   *Result:* false
"""

GEMINI_MODEL="gemini-1.5-flash-latest"

def init_gemini_custom_prompt():
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=[SYSTEM_INSTRUCTIONS],
    )
    return model

"""
Check if the user speech includes a request for one of the supported actions.
@:returns the wanted action of "false" if no action was requested.
"""
def check_for_video_action(user_speech: str) -> str:
    model = init_gemini_custom_prompt()
    response = model.generate_content([user_speech])
    return response.text