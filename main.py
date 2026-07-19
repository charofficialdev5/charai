import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from groq import Groq

app = FastAPI()

# Enable CORS so your GitHub Pages website can securely talk to your Render server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom system prompt to give your AI a premium, high-tier personality
SPECIALIZED_SYSTEM_PROMPT = "You are a premium, high-tier intelligence engine. Provide incredibly insightful, professional answers."

class PremiumChatRequest(BaseModel):
    prompt: str
    model_choice: str
    user_gemini_key: str = None
    user_groq_key: str = None

@app.post("/chat")
async def chat_endpoint(request: PremiumChatRequest):
    user_input = request.prompt
    provider = request.model_choice.lower()

    # --- GEMINI SYSTEM ROUTING ---
    if provider == "gemini":
        # Check if the user passed a key from the frontend; if not, check Render's environment
        api_key = request.user_gemini_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="No Gemini API Key supplied by interface context.")
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"{SPECIALIZED_SYSTEM_PROMPT}\n\nUser Question: {user_input}"
            )
            return {"answer": response.text}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # --- GROQ SYSTEM ROUTING ---
    elif provider == "groq":
        # Check if the user passed a key from the frontend; if not, check Render's environment
        api_key = request.user_groq_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="No Groq API Key supplied by interface context.")
        try:
            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SPECIALIZED_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input}
                ]
            )
            return {"answer": completion.choices[0].message.content}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            
    else:
        raise HTTPException(status_code=400, detail="Unknown runtime machine layer targeted.")
