from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()


def get_tags(notes: str, valid_tags: list):
    client = OpenAI(base_url=os.environ.get("LLM_BASE_URL"),
                    api_key=os.environ.get("LLM_API_KEY"))

    system = f"""You translate a user's description of a game they want into tags.

Rules:
- Choose tags ONLY from this exact list. Do not invent tags or reword them:
{valid_tags}
- "want" = tags matching what they want. "avoid" = tags for what they explicitly don't want.
- Use the exact spelling from the list (e.g. "fast_paced", not "fast-paced").
- If the user references a specific game, infer its tags and translate accordingly.
- Pick at most 6 want tags and 4 avoid tags. Fewer is fine.

Return ONLY a JSON object, nothing else:
{{"want": ["tag1", "tag2"], "avoid": ["tag3"]}}"""

    resp = client.chat.completions.create(
        model=os.environ.get("LLM_MODEL"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": notes},
        ],
        response_format={"type": "json_object"},
    )

    raw = json.loads(resp.choices[0].message.content)
    valid = set(valid_tags)
    return {
        "want":  [t for t in raw.get("want", []) if t in valid],
        "avoid": [t for t in raw.get("avoid", []) if t in valid],
    }
