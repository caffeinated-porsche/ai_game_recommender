from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()


def get_tags(notes: str, valid_tags: list):

    # TODO: message = prompt engineer around notes
    LLM_BASE_URL = os.environ.get("LLM_BASE_URL")
    LLM_API_KEY = os.environ.get("LLM_API_KEY")
    LLM_MODEL = os.environ.get("LLM_MODEL")
    client = OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': notes,
            }
        ],
        model=LLM_MODEL,
    )
    # print(chat_completion.choices[0].message.content)
