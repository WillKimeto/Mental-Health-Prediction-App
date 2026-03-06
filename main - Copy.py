"""
Tumaini — Mental Health Prediction App
FastAPI Backend
=======================================
Endpoints:
  POST /predict   — XGBoost risk prediction
  POST /chat      — Groq Llama 3.3 chatbot (streaming)

Setup:
  pip install fastapi uvicorn xgboost scikit-learn numpy pandas groq

Run locally:
  uvicorn main:app --reload

Deploy to Render:
  - Set GROQ_API_KEY as an environment variable in Render dashboard
  - Start command: uvicorn main:app --host 0.0.0.0 --port 10000
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from groq import Groq

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# Locally: set in your terminal with: set GROQ_API_KEY=your-key-here
# On Render: add as an Environment Variable in the dashboard
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_super_secret_api_key")

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(title="Tumaini API")

# Allow your frontend (GitHub Pages) to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten this to your GitHub Pages URL after deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# MODEL + ENCODERS
# ─────────────────────────────────────────────
model = xgb.Booster()
model.load_model("xgboost_model.json")

le_family_history = LabelEncoder()
le_family_history.classes_ = np.array(["No", "Yes"])

le_days_indoors = LabelEncoder()
le_days_indoors.classes_ = np.array(["1-14 days", "15-30 days", "More than 2 months"])

le_growing_stress = LabelEncoder()
le_growing_stress.classes_ = np.array(["Maybe", "No", "Yes"])

# ─────────────────────────────────────────────
# CHATBOT SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are Tumaini (Swahili for "hope"), a compassionate and knowledgeable mental health support assistant built into a Mental Health Risk Prediction App. Your role is to:

1. Help users understand what their risk score means in plain language.
2. Provide general, evidence-based mental health information and coping strategies.
3. Encourage users to seek professional help when appropriate.
4. Be warm, non-judgmental, and supportive at all times.

Important boundaries:
- You are NOT a therapist or doctor. Always clarify this when relevant.
- Never diagnose. Never prescribe. Always recommend professional consultation for serious concerns.
- If a user expresses thoughts of self-harm or suicide, immediately and clearly direct them to a crisis line (e.g., 988 Suicide & Crisis Lifeline in the US, or their local equivalent).
- Keep responses concise and readable — avoid overwhelming the user with text.

The app predicts mental health treatment likelihood using three factors: family history of mental illness, days spent indoors, and growing stress levels. If the user shares their score, use it to give more personalised context."""

# ─────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    family_history: str
    days_indoors: str
    growing_stress: str

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Tumaini API is running"}


@app.post("/predict")
def predict(req: PredictRequest):
    input_data = pd.DataFrame({
        "family_history": [le_family_history.transform([req.family_history])[0]],
        "Days_Indoors":   [le_days_indoors.transform([req.days_indoors])[0]],
        "Growing_Stress": [le_growing_stress.transform([req.growing_stress])[0]],
    }).astype(float)

    dmatrix = xgb.DMatrix(input_data)
    probability = float(model.predict(dmatrix)[0])

    return {
        "probability": probability,
        "percentage": f"{probability:.2%}",
        "high_risk": probability > 0.5
    }

@app.post("/chat")
def chat(req: ChatRequest):
    client = Groq(api_key=GROQ_API_KEY)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in req.history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": req.message})

    def stream_response():
        with client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1024,
            stream=True,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    return StreamingResponse(stream_response(), media_type="text/plain")