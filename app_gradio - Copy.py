import gradio as gr
import xgboost as xgb
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — required for Gradio
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from groq import Groq

GROQ_API_KEY = "your_super_secret_api_key_here"  # Replace with your actual key from https://console.groq.com

model = xgb.Booster()
model.load_model("xgboost_model.json")

# Label encoders
le_family_history = LabelEncoder()
le_family_history.classes_ = np.array(["No", "Yes"])

le_days_indoors = LabelEncoder()
le_days_indoors.classes_ = np.array(["1-14 days", "15-30 days", "More than 2 months"])

le_growing_stress = LabelEncoder()
le_growing_stress.classes_ = np.array(["Maybe", "No", "Yes"])

# Prediction function
def predict(family_history, days_indoors, growing_stress):
    input_data = pd.DataFrame({
        "family_history": [le_family_history.transform([family_history])[0]],
        "Days_Indoors":   [le_days_indoors.transform([days_indoors])[0]],
        "Growing_Stress": [le_growing_stress.transform([growing_stress])[0]],
    })

    dmatrix = xgb.DMatrix(input_data)
    probability = model.predict(dmatrix)[0]
    pct = f"{probability:.2%}"

    if probability > 0.5:
        result = (
            f"## 🔴 Risk Score: {pct}\n\n"
            "**Prediction: Likely to benefit from treatment (Yes)**\n\n"
            "Based on your inputs, the model flags an elevated mental health risk. "
            "This does not mean you have a condition — it means these patterns are "
            "associated with individuals who have sought treatment. "
            "Consider speaking to a mental health professional for a proper assessment.\n\n"
            "> 💬 Head to the **Tumaini** tab to ask follow-up questions."
        )
    else:
        result = (
            f"## 🟢 Risk Score: {pct}\n\n"
            "**Prediction: Lower likelihood of needing treatment (No)**\n\n"
            "Your inputs suggest a lower risk profile based on the model's patterns. "
            "Mental health is dynamic — continue to check in with yourself regularly.\n\n"
            "> 💬 Head to the **Tumaini** tab if you have any questions."
        )

    # SHAP — runs for BOTH high and low risk (fixed: was accidentally inside the else block)
    fig, ax = plt.subplots(figsize=(6, 3))
    try:
        input_data_float = input_data.astype(float)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data_float)
        shap.summary_plot(shap_values, input_data_float, plot_type="bar", show=False)
        fig = plt.gcf()
    except Exception as e:
        ax.text(0.5, 0.5, f"SHAP unavailable:\n{e}", ha="center", va="center")
        ax.axis("off")

    return result, fig

# Chatbot system prompt
SYSTEM_PROMPT = """You are a compassionate and knowledgeable mental health support assistant built into a Mental Health Risk Prediction App. Your role is to:

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


def chat(user_message, history):
    if not GROQ_API_KEY or GROQ_API_KEY == "your-groq-api-key-here":
        yield "⚠️ **No API key set.** Open `app_gradio.py` and replace `your-groq-api-key-here` with your key from https://console.groq.com"
        return

    client = Groq(api_key=GROQ_API_KEY)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1024,
        stream=True,
    )

    response_text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            response_text += delta
            yield response_text

# CSS


# ── UI 
# gr.Blocks() has NO css= argument in Gradio 6 — css goes in launch() below
with gr.Blocks(title="Mental Health Prediction App") as app:

    gr.Markdown(
        """
        # Mental Health Treatment Prediction App

        ---
        > ⚠️ **Disclaimer:** This tool is for educational purposes only and does not constitute
        > medical advice. If you are concerned about your mental health, please consult a qualified professional.
        """
    )

    # gr.Tabs() must be INSIDE the with gr.Blocks() block
    with gr.Tabs():

        # Tab 1 — Prediction
        with gr.Tab("Risk Prediction"):

            gr.Markdown(
                """
                ### How it works
                Answer three questions about your background and lifestyle.
                The model will estimate your likelihood of benefiting from mental health treatment,
                and explain which factors drove the prediction using SHAP analysis.
                """
            )

            with gr.Row():
                with gr.Column():
                    family_history_input = gr.Dropdown(
                        choices=["Yes", "No"],
                        label="Family History of Mental Health Issues",
                        info="Has anyone in your immediate family been diagnosed with or treated for a mental health condition?",
                        value="No",
                    )
                    days_indoors_input = gr.Dropdown(
                        choices=["1-14 days", "15-30 days", "More than 2 months"],
                        label="Days Indoors (over the past few months)",
                        info="On average, how many days have you been staying indoors?",
                        value="1-14 days",
                    )
                    growing_stress_input = gr.Dropdown(
                        choices=["Yes", "No", "Maybe"],
                        label="Growing Stress",
                        info="Have you noticed your stress levels increasing recently?",
                        value="No",
                    )
                    predict_btn = gr.Button("🔍 Predict", variant="primary")

                with gr.Column():
                    result_output = gr.Markdown(label="Prediction Result")
                    shap_output = gr.Plot(label="SHAP Feature Importance")

            predict_btn.click(
                fn=predict,
                inputs=[family_history_input, days_indoors_input, growing_stress_input],
                outputs=[result_output, shap_output],
            )

        # Tab 2 — Chatbot
        with gr.Tab("Tumaini Mental Health Chatbot"):

            gr.Markdown(
                """
                ### Chat with Tumaini your Mental Health Assistant
                Ask anything about your prediction score, mental health in general,
                coping strategies, or when to seek professional help.
                The assistant is powered by Llama 3.3 (via Groq) and is context-aware of this app.
                """
            )

            gr.ChatInterface(
                fn=chat,
                type="messages",
                chatbot=gr.Chatbot(
                    height=450,
                    type="messages",
                    placeholder="👋 Hi! I'm Tumaini, your mental health support assistant. Ask me anything about your results or mental health in general.",
                ),
                textbox=gr.Textbox(
                    placeholder="e.g. What does a 70% risk score mean?  |  What are some stress management tips?",
                    container=False,
                    scale=7,
                ),
                submit_btn="Send ➤",
                
                examples=[
                    "What does my risk score actually mean?",
                    "What are some simple techniques to manage growing stress?",
                    "When should someone consider seeing a therapist?",
                    "How does family history affect mental health risk?",
                    "I feel anxious a lot — is that normal?",
                ],
            )

# Launch — css goes here in Gradio 6
if __name__ == "__main__":
    app.launch(
        share=False,
        show_error=True,
    )