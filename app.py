import os
import csv
import time
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
import plotly.graph_objects as go

# Import our OSC receiver module 
import osc_receiver

# Start OSC Server 
if "osc_server_started" not in st.session_state:
    st.session_state.osc_server = osc_receiver.start_osc_server(ip="0.0.0.0", port=5000)
    st.session_state.osc_server_started = True

# Participant ID Input 
if "participant_id" not in st.session_state:
    st.session_state.participant_id = ""

# On the Home page, prompt for Participant ID if not set
page = st.sidebar.radio("Navigate", ["Home", "Stages"])

if page == "Home":
    st.title("Welcome to the AI Hiring Assistant Study")
    st.markdown("""
    **About the Study:**

    This study explores trust in AI within the context of recruitment. An AI Hiring Assistant screens candidates and predicts whether they are a good fit for the role.
    
    Please enter your Participant ID below to begin.
    """)
    participant_input = st.text_input("Participant ID:", st.session_state.participant_id)
    if participant_input:
        st.session_state.participant_id = participant_input
        # Update OSC receiver with participant ID so EEG files are named accordingly
        osc_receiver.participant_id = st.session_state.participant_id
        st.success(f"Participant ID set to: {st.session_state.participant_id}")
    else:
        st.warning("Please enter your Participant ID to continue.")
        st.stop()

# Function to record decisions to CSV
def record_decision(decision, candidate_info, stage):
    participant_id = st.session_state.participant_id
    filename = f"user_decisions_{participant_id}.csv"
    timestamp = time.time()
    row = [
        timestamp,
        stage,
        st.session_state.candidate_idx + 1,
        candidate_info.get("Gender", ""),
        candidate_info.get("Experience", ""),
        candidate_info.get("Test_Score", ""),
        candidate_info.get("Interview_Score", ""),
        decision
    ]
    file_exists = os.path.exists(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "stage", "candidate_index", "Gender", "Experience", "Test_Score", "Interview_Score", "decision"])
        writer.writerow(row)

# Sidebar Navigation 
if page == "Home":
    st.info("Please use the sidebar to navigate to 'Stages' once your Participant ID is set.")
    st.stop()

#  Main Study Page 
st.title("AI Hiring Assistant Study")
st.markdown("""
In this study, an AI Hiring Assistant screens candidates for a role and predicts whether they are a good fit.
Please evaluate the predictions and share your trust in the system.
""")

# Stage Selection 
stage = st.radio("Select Stage", options=["Stage 1", "Stage 2", "Stage 3", "Stage 4"])

# Reset Candidate & Prediction State if Stage Changes
if "prev_stage" not in st.session_state:
    st.session_state.prev_stage = stage
elif st.session_state.prev_stage != stage:
    st.session_state.candidate_idx = 0
    st.session_state.prediction_made = False
    st.session_state.current_prediction = {}
    st.session_state.prev_stage = stage

# Update OSC receiver's global variables so EEG data are recorded with the correct stage and participant ID
osc_receiver.current_stage = stage
osc_receiver.participant_id = st.session_state.participant_id

# Load Model and Scaler Based on Stage
if stage in ["Stage 1", "Stage 2"]:
    model_path = "assistant/models/biased_logistic_model.pkl"
    scaler_path = "assistant/models/logistic_scaler.pkl"
    st.write("Loading Model...")
else:
    model_path = "assistant/models/fair_logistic_model.pkl"
    scaler_path = "assistant/models/fair_logistic_scaler.pkl"
    st.write("Loading Model...")

try:
    model = load(model_path)
    scaler = load(scaler_path)
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Define Candidate Data for Each Stage 
if stage == "Stage 1":
    candidate_data = {
        "Gender_binary": [1, 0, 1, 0],
        "Experience": [6, 6, 8, 8],
        "Test_Score": [50, 50, 90, 90],
        "Interview_Score": [7, 7, 9, 9],
    }
elif stage == "Stage 2":
    candidate_data = {
        "Gender_binary": [1, 0, 1, 0],
        "Experience": [6, 6, 8, 8],
        "Test_Score": [50, 50, 90, 90],
        "Interview_Score": [7, 7, 9, 9],
    }
elif stage == "Stage 3":
    candidate_data = {
      "Gender_binary": [1, 0, 1, 0],
        "Experience": [6, 6, 8, 8],
        "Test_Score": [50, 50, 90, 90],
        "Interview_Score": [7, 7, 9, 9],
    }
elif stage == "Stage 4":
    candidate_data = {
      "Gender_binary": [1, 0, 1, 0],
        "Experience": [6, 6, 8, 8],
        "Test_Score": [50, 50, 90, 90],
        "Interview_Score": [7, 7, 9, 9],
    }

candidate_df = pd.DataFrame(candidate_data)
candidate_df["Gender"] = candidate_df["Gender_binary"].map({1: "Male", 0: "Female"})
candidate_df = candidate_df.head(4)

# Initialize Session State Variables 
if "candidate_idx" not in st.session_state:
    st.session_state.candidate_idx = 0
if "responses" not in st.session_state:
    st.session_state.responses = {}
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = {}

# Candidate Navigation Functions 
def prev_candidate():
    st.session_state.candidate_idx = max(0, st.session_state.candidate_idx - 1)
    st.session_state.prediction_made = False
    st.session_state.current_prediction = {}

def next_candidate():
    st.session_state.candidate_idx = min(len(candidate_df) - 1, st.session_state.candidate_idx + 1)
    st.session_state.prediction_made = False
    st.session_state.current_prediction = {}

# Display Current Candidate with Visual Feature Values 
current_candidate = candidate_df.iloc[st.session_state.candidate_idx]
st.markdown(f"### Candidate {st.session_state.candidate_idx + 1}")

# Layout columns for each feature
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write("**Gender**")
    st.write(current_candidate['Gender'])
with col2:
    st.write("**Experience**")
    st.write(f"{current_candidate['Experience']} / 10")
    st.progress(current_candidate['Experience'] / 10)
with col3:
    st.write("**Test Score**")
    st.write(f"{current_candidate['Test_Score']} / 100")
    st.progress(current_candidate['Test_Score'] / 100)
with col4:
    st.write("**Interview Score**")
    st.write(f"{current_candidate['Interview_Score']} / 10")
    st.progress(current_candidate['Interview_Score'] / 10)

col_nav1, col_nav2 = st.columns(2)
col_nav1.button("Previous Candidate", on_click=prev_candidate)
col_nav2.button("Next Candidate", on_click=next_candidate)

# Prediction and Response Section 
if st.button("Predict Hiring Outcome for This Candidate"):
    X_candidate = np.array([[current_candidate['Experience'],
                              current_candidate['Test_Score'],
                              current_candidate['Interview_Score'],
                              current_candidate['Gender_binary']]])
    X_candidate_scaled = scaler.transform(X_candidate)
    prob = model.predict_proba(X_candidate_scaled)[0, 1]
    prediction = "Hired" if prob >= 0.5 else "Not Hired"
    st.session_state.current_prediction = {
        "prob": np.round(prob, 3),
        "prediction": prediction,
        "X_candidate_scaled": X_candidate_scaled
    }
    st.session_state.prediction_made = True

if st.session_state.prediction_made:
    st.markdown("#### Prediction")
    predicted_prob = st.session_state.current_prediction['prob']
    st.write(f"Predicted Hiring Probability: {int(predicted_prob * 100)}%")
    st.progress(predicted_prob)
    
    # Display the prediction in bold and colored (green for Hired, red for Not Hired)
    prediction_text = st.session_state.current_prediction['prediction']
    color = "green" if prediction_text == "Hired" else "red"
    st.markdown(f"<h3 style='color: {color}; font-weight: bold;'>{prediction_text}</h3>", unsafe_allow_html=True)
    
    # Enhanced SHAP Explanation 
    if stage in ["Stage 2", "Stage 4"]:
        st.markdown("#### SHAP Explanation")
        if hasattr(model, "calibrated_classifiers_"):
            model_for_explainer = model.calibrated_classifiers_[0].estimator
        elif hasattr(model, "base_estimator"):
            model_for_explainer = model.base_estimator
        else:
            model_for_explainer = model

        explainer = shap.LinearExplainer(
            model_for_explainer,
            scaler.transform(candidate_df[["Experience", "Test_Score", "Interview_Score", "Gender_binary"]])
        )
        shap_values = explainer.shap_values(st.session_state.current_prediction["X_candidate_scaled"])
        candidate_shap = shap_values[0]
        feature_names = ["Experience", "Test_Score", "Interview_Score", "Gender_binary"]

        # Create interactive horizontal bar chart using Plotly
        colors = ['green' if val >= 0 else 'red' for val in candidate_shap]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=candidate_shap,
            y=feature_names,
            orientation='h',
            marker_color=colors,
            hovertemplate='%{y}: %{x:.2f}<extra></extra>'
        ))
        fig.update_layout(
            title="Feature Contributions",
            xaxis_title="Contribution to Prediction",
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig, use_container_width=True)

        # Create a natural language summary with bullet points
        sorted_idx = np.argsort(np.abs(candidate_shap))[::-1]
        summary_lines = []
        for i in sorted_idx:
            feature = feature_names[i]
            val = candidate_shap[i]
            direction = "increases" if val >= 0 else "decreases"
            summary_lines.append(f"- **{feature}** {direction} the hiring probability by {abs(val):.2f}.")
        summary_text = "\n".join(summary_lines)
        st.markdown(f"**Summary:**\n{summary_text}")
    
    col_resp1, col_resp2 = st.columns(2)
    if col_resp1.button("Accept & Trust", key=f"accept_{st.session_state.candidate_idx}"):
        st.session_state.responses[f"Candidate_{st.session_state.candidate_idx + 1}"] = "Accept & Trust"
        st.success("Response recorded: Accept & Trust")
        # Record decision to CSV
        record_decision("Accept & Trust", current_candidate.to_dict(), stage)
    if col_resp2.button("Reject & Distrust", key=f"reject_{st.session_state.candidate_idx}"):
        st.session_state.responses[f"Candidate_{st.session_state.candidate_idx + 1}"] = "Reject & Distrust"
        st.success("Response recorded: Reject & Distrust")
        # Record decision to CSV
        record_decision("Reject & Distrust", current_candidate.to_dict(), stage)

st.markdown("### Your Responses")
if st.session_state.responses:
    st.json(st.session_state.responses)
else:
    st.info("No responses recorded yet.")
