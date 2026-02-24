import streamlit as st
import pickle as pickle
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load model
with open("../model/my_data.pkl", "rb") as f:
    model = pickle.load(f)

scaler = pickle.load(open("../model/scaler.pkl", "rb"))

# Load features
data = pd.read_csv("../model/features.csv")

def get_scaled_values(input_dict):
    X = data.drop(['diagnosis'], axis=1, errors='ignore')  # drop diagnosis if present

    scaler = MinMaxScaler()
    scaler.fit(X)

    input_df = pd.DataFrame([input_dict])
    scaled_array = scaler.transform(input_df)
    scaled_dict = dict(zip(input_df.columns, scaled_array[0]))
    return scaled_dict

def add_asidebar():
    st.sidebar.header("Input Cell Nuclei Measurements")

    slider_labels = [
        ("Radius 1", "radius1"),
        ("Texture 1", "texture1"),
        ("Perimeter 1", "perimeter1"),
        ("Area 1", "area1"),
        ("Smoothness 1", "smoothness1"),
        ("Compactness 1", "compactness1"),
        ("Concavity 1", "concavity1"),
        ("Concave Points 1", "concave_points1"),
        ("Symmetry 1", "symmetry1"),
        ("Fractal Dimension 1", "fractal_dimension1"),

        ("Radius 2", "radius2"),
        ("Texture 2", "texture2"),
        ("Perimeter 2", "perimeter2"),
        ("Area 2", "area2"),
        ("Smoothness 2", "smoothness2"),
        ("Compactness 2", "compactness2"),
        ("Concavity 2", "concavity2"),
        ("Concave Points 2", "concave_points2"),
        ("Symmetry 2", "symmetry2"),
        ("Fractal Dimension 2", "fractal_dimension2"),

        ("Radius 3", "radius3"),
        ("Texture 3", "texture3"),
        ("Perimeter 3", "perimeter3"),
        ("Area 3", "area3"),
        ("Smoothness 3", "smoothness3"),
        ("Compactness 3", "compactness3"),
        ("Concavity 3", "concavity3"),
        ("Concave Points 3", "concave_points3"),
        ("Symmetry 3", "symmetry3"),
        ("Fractal Dimension 3", "fractal_dimension3"),
    ]

    input_dictionary = {}

    for label, key in slider_labels:
        input_dictionary[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

    return input_dictionary

def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
        input_data['radius1'], input_data['texture1'], input_data['perimeter1'],
        input_data['area1'], input_data['smoothness1'], input_data['compactness1'],
        input_data['concavity1'], input_data['concave_points1'], input_data['symmetry1'],
        input_data['fractal_dimension1']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
        input_data['radius2'], input_data['texture2'], input_data['perimeter2'],
        input_data['area2'], input_data['smoothness2'], input_data['compactness2'],
        input_data['concavity2'], input_data['concave_points2'], input_data['symmetry2'],
        input_data['fractal_dimension2']
    ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
        input_data['radius3'], input_data['texture3'], input_data['perimeter3'],
        input_data['area3'], input_data['smoothness3'], input_data['compactness3'],
        input_data['concavity3'], input_data['concave_points3'], input_data['symmetry3'],
        input_data['fractal_dimension3']
    ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True
    )

    return fig

def make_predictions(input_data):
     model = pickle.load(open("../model/my_data.pkl", "rb"))
     scaler = pickle.load(open("../model/scaler.pkl", "rb"))

     array = np.array(list(input_data.values())).reshape(1, -1)
     #st.write(array)

     scaledarray = scaler.transform(array)

     prediction = model.predict(scaledarray)
     #st.write(prediction)

     if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
     else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
    
  
        st.write(f"Probability of being Benign: {model.predict_proba(scaledarray)[0][0]:.2f}")
        st.write(f"Probability of being Malignant: {model.predict_proba(scaledarray)[0][1]:.2f}")
        st.write("WARNING: This is not a better predictor than an actual medical diagnosis.")
     





def main ():
    st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
  )
    
    userinput = add_asidebar()
    

    #st.write(userinput)

    with st.container():
        st.title("Breast Cancer Prediction")
        st.write("Predict using the sidebars whether a breast cancer is malignant or benign.")

    col1, col2 = st.columns([4,1])

    with col1:
        radar_chart = get_radar_chart(userinput)
        st.plotly_chart(radar_chart)
    with col2:
        st.write("Prediction")
        make_predictions(userinput)
    
    


if __name__ == "__main__":
    main()