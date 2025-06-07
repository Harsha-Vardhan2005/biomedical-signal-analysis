import streamlit as st
import numpy as np
import pandas as pd
from components.EEGModels import EEGNet
from groq import Groq
import os


# Paste your Groq API Key here
GROQ_API_KEY = "your-api-key"
client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="ML Classification", page_icon="🤖")

st.title("EEG Classification with ML and LLM Insights")

uploaded_file = st.file_uploader("Upload EEG CSV File", type=["csv"])
if uploaded_file is not None:
    # Save uploaded file temporarily
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load data
    try:
        data = pd.read_csv(file_path, delimiter=",", header=0, names=["A1", "A2"])
        if data.empty or "A1" not in data.columns or "A2" not in data.columns:
            st.error("Invalid CSV format. Ensure it has 'A1' and 'A2' columns.")
        else:
            st.write(f"Signal shape: {data.shape}")

            # Preprocess data for EEGNet
            signal = np.stack([data["A1"].values, data["A2"].values], axis=0).T
            if len(signal) > 1000:
                signal = signal[:1000]  # Truncate to 1000 samples
            else:
                signal = np.pad(signal, ((0, 1000 - len(signal)), (0, 0)), mode='constant')
            signal = signal.reshape(1, 2, 1000, 1)  # Batch, Channels, Samples, Depth

            # Load trained model
            model = EEGNet(nb_classes=2, Chans=2, Samples=1000, dropoutRate=0.5)
            model.load_weights("eegnet_trained.weights.h5")  # Ensure this file exists
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Predict
            prediction = model.predict(signal)
            class_label = "Seizure" if prediction[0][1] > 0.5 else "Normal"
            confidence = prediction[0][1] if class_label == "Seizure" else prediction[0][0]
            st.write(f"Prediction: {class_label} (Confidence: {confidence:.2f})")

            # LLM Analysis
            prompt = f"""
            You are an expert in EEG signal analysis. Analyze the following EEG classification result:
            - Predicted class: {class_label}
            - Confidence score: {confidence:.2f}

            Provide a clear explanation for a general user about the brain activity, including:
            - What the prediction means (normal or abnormal activity, e.g., seizure).
            - What the confidence score indicates.
            - Any recommendations or next steps (e.g., consult a specialist if abnormal).

            Keep the language simple, informative, and avoid technical jargon where possible.
            """
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )
            st.write("LLM Classification Insight:", response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error processing the uploaded file: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
else:
    st.warning("Please upload a CSV file and ensure your Groq API Key is correctly pasted above to proceed.")