import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def chat_with_ai(prompt):

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return {
            "success": True,
            "text": response.text
        }

    except Exception as e:

        return {
            "success": False,
            "text": f"Gemini Error: {e}"
        }


def generate_ai_investigation(ticket, category, department):

    prompt = f"""
You are an expert IT Support Engineer.

A customer submitted this support ticket.

Ticket:
{ticket}

Predicted Category:
{category}

Assigned Department:
{department}

Analyze the issue and return ONLY using this format:

Possible Root Cause:
...

Recommended Resolution:
- ...
- ...
- ...

Professional Reply:
...
"""

    return chat_with_ai(prompt)