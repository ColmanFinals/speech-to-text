import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import os

SUPPORTED_COMMANDS = os.environ["SUPPORTED_COMMANDS"].split(',')
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
SYSTEM_INSTRUCTIONS = f"""
Given a user's input, classify their intent, to one of the following commands: {SUPPORTED_COMMANDS} if possible else false. 
Answer a single command or false if the command is not comprehensible.

Text: Begin the process.
Answer: start

Text: Cease all operations.
Answer: stop

Text: Do it once more.
Answer: repeat

Text: Let's go again.
Answer: repeat

Text: Initiate the program.
Answer: start

Text: Put everything on hold.
Answer: stop

Text: Hi
Answer: false

Text: mkelvnldr
Answer: false

Text: stump // can be a mistake during translation from speech to text, and sounds similar to start.
Answer: start 

Text: בצלגללםכם
Answer: false
    
Text: Lo siento
Answer: false 
"""

GEMINI_MODEL = "gemini-1.5-flash-latest"


def init_gemini_custom_prompt():
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(
        generation_config=genai.types.GenerationConfig(top_k=1, temperature=0, max_output_tokens=10),
        model_name=GEMINI_MODEL,
        system_instruction=[SYSTEM_INSTRUCTIONS],
    )
    return model


"""
Check if the user speech includes a request for one of the supported actions.
@:returns the wanted action of "false" if no action was requested.
"""


def check_for_video_action(user_speech: str) -> str:
    try:
        model = init_gemini_custom_prompt()
        response = model.generate_content([user_speech])
        return response.text
    except ResourceExhausted:
        print("Gemini api got to quotas limit. Return false")
        return "false"
