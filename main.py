import os
import uuid
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from groq import Groq

app = FastAPI()

# Enable CORS so your GitHub Pages website can securely talk to your Render server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from your GitHub Pages URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session database to track usage limits per user
SESSION_USAGE = {}
MAX_FREE_MESSAGES_PER_SESSION = 30

# Change this text to whatever you want your specialized AI's personality to be!
SPECIALIZED_SYSTEM_PROMPT = "You are a specialized AI assistant. Keep responses brief, helpful, and strictly accurate."

class ChatRequest(BaseModel):
    prompt: str
    model_choice: str

@app.get("/get-session")
async def get_session():
    """Generates an anonymous session ID when the user loads the webpage"""
    session_id = str(uuid.uuid4())
    SESSION_USAGE[session_id] = 0
    return {"session_id": session_id}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, x_session_id: str = Header(None)):
    """Handles routing the prompt to Groq or Gemini based on user selection"""
    if not x_session_id:
        raise HTTPException(status_code=401, detail="Missing session token. Please reload the page.")
    
    if x_session_id not in SESSION_USAGE:
        SESSION_USAGE[x_session_id] = 0

    # Guardrail to protect your budget/rate limits
    if SESSION_USAGE[x_session_id] >= MAX_FREE_MESSAGES_PER_SESSION:
        raise HTTPException(status_code=429, detail="Free message limit reached for this session.")

    user_input = request.prompt
    provider = request.model_choice.lower()

    # Increment usage counter
    SESSION_USAGE[x_session_id] += 1
    remaining_messages = MAX_FREE_MESSAGES_PER_SESSION - SESSION_USAGE[x_session_id]

    # --- GOOGLE GEMINI LAYER ---
    if provider == "gemini":
        try:
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"{SPECIALIZED_SYSTEM_PROMPT}\n\nUser Question: {user_input}"
            )
            return {"answer": response.text, "remaining": remaining_messages}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

    # --- GROQ LAYER ---
    elif provider == "groq":
        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SPECIALIZED_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input}
                ]
            )
            return {"answer": completion.choices[0].message.content, "remaining": remaining_messages}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Groq error: {str(e)}")
            
    else:
        raise HTTPException(status_code=400, detail="Invalid engine choice.")
