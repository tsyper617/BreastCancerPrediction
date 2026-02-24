# Breast Cancer Prediction - Full ML Pipeline
# Imports

import pandas as pd
import pickle 
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def fetch_data():
    dataset = fetch_ucirepo(id=17)

    X = dataset.data.features
    y = dataset.data.targets

    return X, y

def clean_data(X, y):

    df = pd.concat([X, y], axis=1)

    # Convert Diagnosis to numeric
    df['Diagnosis'] = df['Diagnosis'].map({ 'M': 1, 'B': 0    
    })

    # Check for missing values
    if df.isnull().sum().any():
        raise ValueError("Missing values detected")

    X_clean = df.drop('Diagnosis', axis=1)
    y_clean = df['Diagnosis']

    return X_clean, y_clean

def split_and_scale(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):

    model = LogisticRegression()

    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Model Accuracy:", accuracy)

def main():

    print("Fetching data...")
    X, y = fetch_data()

    print("Cleaning data...")
    X, y = clean_data(X, y)

    print("Splitting and scaling...")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    # Save the cleaned features for your app
    X.to_csv("features.csv", index=False)

    # Save object to file
    with open("my_data.pkl", "wb") as f:
        pickle.dump(model, f)

# Load object from file
    with open("my_data.pkl", "rb") as f:
        loaded_object = pickle.load(f)  

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


# Run Program

    print("Feature labels:")
    feature_labels = X.columns.tolist()
    print(feature_labels)


if __name__ == "__main__":
    main()