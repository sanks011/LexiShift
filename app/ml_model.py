import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_user_data():
    """Loads user feedback data, handling missing or empty files gracefully."""
    file_path = 'user_feedback.csv'
    
    if not os.path.exists(file_path):
        print("Warning: user_feedback.csv not found. Using default settings.")
        return None  # Handle missing data scenario
    
    data = pd.read_csv(file_path)
    
    if data.empty:
        print("Warning: user_feedback.csv is empty. Using default settings.")
        return None
    
    return data

def train_model():
    """Trains a KMeans clustering model based on user feedback."""
    data = load_user_data()
    if data is None:
        return None  # Skip training if no data is available

    # Ensure correct types
    data[['font_size', 'line_spacing', 'letter_spacing']] = data[['font_size', 'line_spacing', 'letter_spacing']].apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)  # Remove rows with NaN values

    # Define features
    X = data[['font_name', 'font_size', 'line_spacing', 'letter_spacing', 'text_color']]
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['font_size', 'line_spacing', 'letter_spacing']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['font_name', 'text_color'])
        ]
    )

    # KMeans Model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('kmeans', KMeans(n_clusters=3, random_state=42))  # Adjust clusters dynamically
    ])

    model.fit(X)
    return model

def predict_formatting(user_preferences, model):
    """Predicts the best formatting settings based on user preferences."""
    if model is None:
        return {}  # No trained model available

    prediction = model.predict(user_preferences)

    # Define formatting recommendations based on cluster
    recommendations = {
        0: {'font_name': 'Arial', 'font_size': 12, 'line_spacing': 14, 'letter_spacing': 0.1, 'text_color': '#333333'},
        1: {'font_name': 'OpenDyslexic', 'font_size': 14, 'line_spacing': 16, 'letter_spacing': 0.2, 'text_color': '#000000'},
        2: {'font_name': 'Verdana', 'font_size': 13, 'line_spacing': 15, 'letter_spacing': 0.15, 'text_color': '#111111'}
    }

    return recommendations.get(prediction[0], {})  # Default to empty dict
