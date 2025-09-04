import requests
import json
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

app = Flask(__name__)

# --- Utility functions ---

def getKey():
    """
    Returns API key for the LLM from environment variable
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("❌ Missing GROQ_API_KEY in environment variables")
    return api_key


def getBaseUrl():
    """
    Returns base URL for the LLM from environment variable
    """
    return os.getenv("BASE_URL", "https://api.groq.com/openai/v1")


# --- Prompts (unchanged) ---

def story_system_prompt():
    return """ ... """  # keep your full prompt here


def title_system_prompt():
    return """ ... """  # keep your full prompt here


# --- LLM Call ---

def callLLM(messages):
    """
    Returns output from LLM call
    """
    API_KEY = getKey()
    BASE_URL = getBaseUrl()
    API_URL = f"{BASE_URL}/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 2000,
        "top_p": 1.0,
        "stream": False,
        "stop": None,
    }

    response = requests.post(API_URL, headers=headers, json=data)
    response.raise_for_status()
    output = response.json()["choices"][0]["message"]["content"]

    return output


# --- Routes ---

@app.route("/get_titles", methods=["GET", "PUT"])
def generate_titles():
    title = request.args.get("title") or request.json.get("title")
    description = request.args.get("description") or request.json.get("description")

    messages = [
        {"role": "system", "content": title_system_prompt()},
        {"role": "user", "content": f"user_title: {title} \n user_description: {description}"},
    ]

    response = callLLM(messages)
    return jsonify(json.loads(response)), 200


@app.route("/get_stories", methods=["GET", "PUT"])
def generate_stories():
    title = request.args.get("title") or request.json.get("title")
    description = request.args.get("description") or request.json.get("description")

    messages = [
        {"role": "system", "content": story_system_prompt()},
        {"role": "user", "content": f"user_title: {title} \n user_description: {description}"},
    ]

    response = callLLM(messages)
    return jsonify(json.loads(response)), 200


# --- Run Server ---

if __name__ == "__main__":
    # Optional: allow binding host from env (useful for Docker/Cloud)
    FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
    FLASK_PORT = int(os.getenv("FLASK_PORT", 8000))

    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)
