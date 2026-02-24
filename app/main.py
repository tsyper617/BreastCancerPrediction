# app/main.py

import os
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Paths for model and features
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = os.path.join(PROJECT_ROOT, "model", "my_data.pkl")
SCALER_FILE = os.path.join(PROJECT_ROOT, "model", "scaler.pkl")
FEATURES_FILE = os.path.join(PROJECT_ROOT, "model", "features.csv")

# -----------------------------
# Load model, scaler, and data
# -----------------------------
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

with open(SCALER_FILE, "rb") as f:
    scaler = pickle.load(f)

data = pd.read_csv(FEATURES_FILE)

# -----------------------------
# Helper functions
# -----------------------------

def get_scaled_values(input_dict):
    """
    Scales user input using the loaded MinMaxScaler fitted on training data.
    """
    X = data.drop(['Diagnosis'], axis=1, errors='ignore')
    scaler_local = MinMaxScaler()
    scaler_local.fit(X)

    input_df = pd.DataFrame([input_dict])
    scaled_array = scaler_local.transform(input_df)
    return dict(zip(input_df.columns, scaled_array[0]))

def add_sidebar():
    """
    Adds the Streamlit sidebar with sliders for all features.
    """
    st.sidebar.header("Input Cell Nuclei Measurements")

    slider_labels = [
        ("Radius 1", "radius1"), ("Texture 1", "texture1"), ("Perimeter 1", "perimeter1"),
        ("Area 1", "area1"), ("Smoothness 1", "smoothness1"), ("Compactness 1", "compactness1"),
        ("Concavity 1", "concavity1"), ("Concave Points 1", "concave_points1"),
        ("Symmetry 1", "symmetry1"), ("Fractal Dimension 1", "fractal_dimension1"),
        ("Radius 2", "radius2"), ("Texture 2", "texture2"), ("Perimeter 2", "perimeter2"),
        ("Area 2", "area2"), ("Smoothness 2", "smoothness2"), ("Compactness 2", "compactness2"),
        ("Concavity 2", "concavity2"), ("Concave Points 2", "concave_points2"),
        ("Symmetry 2", "symmetry2"), ("Fractal Dimension 2", "fractal_dimension2"),
        ("Radius 3", "radius3"), ("Texture 3", "texture3"), ("Perimeter 3", "perimeter3"),
        ("Area 3", "area3"), ("Smoothness 3", "smoothness3"), ("Compactness 3", "compactness3"),
        ("Concavity 3", "concavity3"), ("Concave Points 3", "concave_points3"),
        ("Symmetry 3", "symmetry3"), ("Fractal Dimension 3", "fractal_dimension3")
    ]

    input_dict = {}
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

    return input_dict

def get_radar_chart(input_data):
    """
    Generates a radar chart using Plotly for the 3 sets of measurements.
    """
    input_data = get_scaled_values(input_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area',
                  'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    # Mean Value
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius1'], input_data['texture1'], input_data['perimeter1'],
           input_data['area1'], input_data['smoothness1'], input_data['compactness1'],
           input_data['concavity1'], input_data['concave_points1'], input_data['symmetry1'],
           input_data['fractal_dimension1']],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))

    # Standard Error
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius2'], input_data['texture2'], input_data['perimeter2'],
           input_data['area2'], input_data['smoothness2'], input_data['compactness2'],
           input_data['concavity2'], input_data['concave_points2'], input_data['symmetry2'],
           input_data['fractal_dimension2']],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    # Worst Value
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius3'], input_data['texture3'], input_data['perimeter3'],
           input_data['area3'], input_data['smoothness3'], input_data['compactness3'],
           input_data['concavity3'], input_data['concave_points3'], input_data['symmetry3'],
           input_data['fractal_dimension3']],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True
    )

    return fig

def make_predictions(input_data):
    """
    Makes predictions using the preloaded model and scaler.
    """
    array = np.array(list(input_data.values())).reshape(1, -1)
    scaled_array = scaler.transform(array)
    prediction = model.predict(scaled_array)

    if prediction[0] == 0:
        st.markdown("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='diagnosis malicious'>Malignant</span>", unsafe_allow_html=True)

    st.write(f"Probability of being Benign: {model.predict_proba(scaled_array)[0][0]:.2f}")
    st.write(f"Probability of being Malignant: {model.predict_proba(scaled_array)[0][1]:.2f}")
    st.warning("This is not a substitute for a medical diagnosis.")

# -----------------------------
# Main App
# -----------------------------
def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    user_input = add_sidebar()

    st.title("Breast Cancer Prediction")
    st.write("Predict whether a breast cancer is malignant or benign using the sidebar inputs.")

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(user_input)
        st.plotly_chart(radar_chart)

    with col2:
        st.subheader("Prediction")
        make_predictions(user_input)


if __name__ == "__main__":
    main()