import streamlit as st
import numpy as np
import os
from components.data_loader import EEGDataLoader
from components.visualizer import EEGVisualizer
from components.wavelet_analyzer import EEGWaveletAnalyzer
from components.entropy_analyzer import EntropyAnalyzer
from components.complexity_analyzer import ComplexityAnalyzer
from groq import Groq

# Paste your Groq API Key here (replace "your-groq-api-key-here" with your actual key)
GROQ_API_KEY = "your-api-key"
client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="LLM Integrated Insights", page_icon="🧠")

st.title("EEG Analysis with Dynamic LLM Insights")

# Upload CSV file
uploaded_file = st.file_uploader("Upload EEG CSV File", type=["csv"])
if uploaded_file is not None:
    # Load data
    eeg_loader = EEGDataLoader(uploaded_file.name)
    data = eeg_loader.load_data()
    if data is not None:
        st.write(f"Signal shape: {data.shape}")
        st.line_chart(data)  # Basic visualization

        # Extract signals
        signal_fp1 = data["A1"].values
        signal_fp2 = data["A2"].values
        sampling_rate = 1000
        total_duration = len(data) / sampling_rate

        # Add slider for dynamic time range
        time_range = st.slider(
            "Select Time Range to Analyze (seconds)",
            0.0, total_duration, (0.0, min(5.0, total_duration)), 0.1
        )

        # EEG Visualization
        visualizer = EEGVisualizer(data, sampling_rate)
        visualizer.plot_channels("A1", "A2", time_range)
        mean_amplitude_fp1 = np.mean(signal_fp1[int(time_range[0] * sampling_rate):int(time_range[1] * sampling_rate)]).tolist()
        mean_amplitude_fp2 = np.mean(signal_fp2[int(time_range[0] * sampling_rate):int(time_range[1] * sampling_rate)]).tolist()

        # LLM Analysis for Visualization
        vis_prompt = f"""
        You are an expert in EEG signal analysis. Analyze the following EEG visualization features:
        - Mean amplitude for Fp1: {mean_amplitude_fp1}
        - Mean amplitude for Fp2: {mean_amplitude_fp2}
        - Time range analyzed: {time_range} seconds
        - Sampling rate: {sampling_rate} Hz

        Provide a clear explanation for a general user about the brain activity based on the visualization, including:
        - Whether the signal suggests normal activity or potential abnormalities.
        - What the mean amplitude indicates about brain states.

        Keep the language simple, informative, and avoid technical jargon where possible.
        """
        vis_response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": vis_prompt}],
            temperature=0.7,
            max_tokens=300
        )
        st.write("LLM Visualization Insight:", vis_response.choices[0].message.content)

        # Frequency Analysis
        wavelet_analyzer = EEGWaveletAnalyzer(signal_fp1, sampling_rate)
        start_idx = int(time_range[0] * sampling_rate)
        end_idx = int(time_range[1] * sampling_rate)
        signal_slice = signal_fp1[start_idx:end_idx]
        coefficients, frequencies = wavelet_analyzer.perform_wavelet_transform(time_range)
        freq_power_fp1 = np.mean(np.abs(coefficients), axis=1)[:5].tolist()
        wavelet_analyzer.plot_wavelet_transform(coefficients, frequencies, time_range)

        # LLM Analysis for Frequency
        freq_prompt = f"""
        You are an expert in EEG signal analysis. Analyze the following EEG frequency analysis features:
        - Frequency power (top 5 bands) for Fp1: {freq_power_fp1}
        - Time range analyzed: {time_range} seconds
        - Sampling rate: {sampling_rate} Hz

        Provide a clear explanation for a general user about the brain activity based on the frequency analysis, including:
        - Whether the signal suggests normal activity or potential abnormalities.
        - What the frequency power indicates about brain states.

        Keep the language simple, informative, and avoid technical jargon where possible.
        """
        freq_response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": freq_prompt}],
            temperature=0.7,
            max_tokens=300
        )
        st.write("LLM Frequency Insight:", freq_response.choices[0].message.content)

        # Entropy Analysis
        entropy_analyzer_fp1 = EntropyAnalyzer(signal_fp1[start_idx:end_idx])
        entropies_fp1 = entropy_analyzer_fp1.calculate_entropies_in_windows(window_size_sec=5)
        avg_entropy_fp1 = {metric: np.mean([window[metric] for window in entropies_fp1]) 
                          for metric in ["Shannon", "Approximate", "Sample"]}

        entropy_analyzer_fp2 = EntropyAnalyzer(signal_fp2[start_idx:end_idx])
        entropies_fp2 = entropy_analyzer_fp2.calculate_entropies_in_windows(window_size_sec=5)
        avg_entropy_fp2 = {metric: np.mean([window[metric] for window in entropies_fp2]) 
                          for metric in ["Shannon", "Approximate", "Sample"]}
        visualizer.plot_entropy_over_time(entropies_fp1, entropies_fp2, 5)

        # LLM Analysis for Entropy
        entropy_prompt = f"""
        You are an expert in EEG signal analysis. Analyze the following EEG entropy analysis features:
        - Average entropy (Shannon, Approximate, Sample) for Fp1: {avg_entropy_fp1}
        - Average entropy (Shannon, Approximate, Sample) for Fp2: {avg_entropy_fp2}
        - Time range analyzed: {time_range} seconds
        - Sampling rate: {sampling_rate} Hz

        Provide a clear explanation for a general user about the brain activity based on the entropy analysis, including:
        - Whether the signal suggests normal activity or potential abnormalities.
        - What the entropy values indicate about brain states.

        Keep the language simple, informative, and avoid technical jargon where possible.
        """
        entropy_response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": entropy_prompt}],
            temperature=0.7,
            max_tokens=300
        )
        st.write("LLM Entropy Insight:", entropy_response.choices[0].message.content)

        # Final Summary
        final_prompt = f"""
        You are an expert in EEG signal analysis. Summarize the following EEG analysis features and provide a final condition:
        - Mean amplitude for Fp1: {mean_amplitude_fp1}
        - Mean amplitude for Fp2: {mean_amplitude_fp2}
        - Frequency power (top 5 bands) for Fp1: {freq_power_fp1}
        - Average entropy (Shannon, Approximate, Sample) for Fp1: {avg_entropy_fp1}
        - Average entropy (Shannon, Approximate, Sample) for Fp2: {avg_entropy_fp2}
        - Time range analyzed: {time_range} seconds
        - Sampling rate: {sampling_rate} Hz

        Provide a clear summary for a general user, including:
        - An overall assessment of brain activity (normal or abnormal).
        - Key observations from amplitude, frequency, and entropy.
        - A final condition or recommendation (e.g., consult a specialist if needed).

        Keep the language simple, informative, and avoid technical jargon where possible.
        """
        final_response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.7,
            max_tokens=500
        )
        st.write("Final LLM Summary:", final_response.choices[0].message.content)
else:
    st.warning("Please upload a CSV file and ensure your Groq API Key is correctly pasted above to proceed.")