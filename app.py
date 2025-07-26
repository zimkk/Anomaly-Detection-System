import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import os
import json
from datetime import datetime


import warnings
warnings.filterwarnings("ignore")

# Global variables to store data and models
training_data = None
testing_data = None
models = {}
preprocessor = None
scaler = StandardScaler()

def initialize_models():
    global models
    models = {}
    
    # Initialize each model individually with error handling
    model_definitions = [
        ("Logistic Regression", lambda: LogisticRegression(max_iter=1000)),
        ("Decision Tree", lambda: DecisionTreeClassifier()),
        ("Random Forest", lambda: RandomForestClassifier()),
        ("K-Nearest Neighbors", lambda: KNeighborsClassifier()),
        ("Gradient Boosting", lambda: GradientBoostingClassifier())
    ]
    
    for model_name, model_creator in model_definitions:
        try:
            models[model_name] = model_creator()
        except Exception as e:
            print(f"Warning: Failed to initialize {model_name}: {e}")
            # Continue with other models even if one fails
    
    return models

def preprocess_data(df):
    # Drop irrelevant columns
    df = df.drop(['id', 'attack_cat'], axis=1, errors='ignore')
    
    # Clamp extreme values
    df_numeric = df.select_dtypes(include=[np.number])
    for feature in df_numeric.columns:
        if df[feature].max() > 10 * df[feature].median():
            df[feature] = np.where(
                df[feature] < df[feature].quantile(0.95),
                df[feature],
                df[feature].quantile(0.95)
            )
    
    # Log transform skewed features
    for feature in df_numeric.columns:
        if df[feature].nunique() > 50:
            df[feature] = np.log1p(df[feature])
    
    # Reduce categorical cardinality
    df_cat = df.select_dtypes(exclude=[np.number])
    for feature in df_cat.columns:
        top_labels = df[feature].value_counts().nlargest(5).index
        df[feature] = df[feature].apply(lambda x: x if x in top_labels else '-')
    
    return df

# The following functions can be called from a PyQt5/Tkinter UI:
# - load_and_preprocess(training_path, testing_path)
# - train_models()
# - stream_next(model_name)
# - analyze_all_models()

def load_and_preprocess(training_path, testing_path):
    global training_data, testing_data
    if not os.path.exists(training_path) or not os.path.exists(testing_path):
        raise FileNotFoundError('One or both file paths do not exist')
    training_data = pd.read_csv(training_path)
    testing_data = pd.read_csv(testing_path)
    training_data = preprocess_data(training_data)
    testing_data = preprocess_data(testing_data)
    return len(training_data), len(testing_data)

def train_models():
    global training_data, testing_data, preprocessor
    if training_data is None or testing_data is None:
        raise ValueError('Data not loaded')
    cat_features = training_data.select_dtypes(include='object').columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ],
        remainder='passthrough'
    )
    initialize_models()
    trained_models = {}
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('scaler', scaler),
            ('classifier', model)
        ])
        pipeline.fit(training_data.drop('label', axis=1), training_data['label'])
        trained_models[name] = pipeline
        joblib.dump(pipeline, f'models/{name.lower().replace(" ", "_")}.joblib')
    return list(models.keys())

def stream_next(model_name):
    global testing_data
    if testing_data is None or len(testing_data) == 0:
        raise ValueError('No more data to stream or data not loaded')
    model = joblib.load(f'models/{model_name.lower().replace(" ", "_")}.joblib')
    row = testing_data.iloc[0:1]
    testing_data = testing_data.iloc[1:]
    prediction = model.predict(row.drop('label', axis=1))
    probability = model.predict_proba(row.drop('label', axis=1))[0][1] if hasattr(model, 'predict_proba') else None
    result = {
        'prediction': int(prediction[0]),
        'probability': float(probability) if probability is not None else None,
        'actual': int(row['label'].iloc[0]),
        'remaining_samples': len(testing_data)
    }
    return result

def stream_next_ensemble(threshold=0.8):
    global testing_data
    if testing_data is None or len(testing_data) == 0:
        raise ValueError('No more data to stream or data not loaded')
    row = testing_data.iloc[0:1]
    testing_data = testing_data.iloc[1:]
    votes = []
    probabilities = {}
    for name in models.keys():
        model = joblib.load(f'models/{name.lower().replace(" ", "_")}.joblib')
        proba = model.predict_proba(row.drop('label', axis=1))[0][1] if hasattr(model, 'predict_proba') else 0
        probabilities[name] = proba
        vote = 1 if proba > threshold else 0
        votes.append(vote)
    # Majority voting
    final_prediction = 1 if sum(votes) > len(votes) // 2 else 0
    result = {
        'prediction': final_prediction,
        'votes': votes,
        'probabilities': probabilities,
        'actual': int(row['label'].iloc[0]),
        'remaining_samples': len(testing_data)
    }
    return result

def analyze_all_models():
    global testing_data
    if testing_data is None:
        raise ValueError('Testing data not loaded')
    results = {}
    X_test = testing_data.drop('label', axis=1)
    y_test = testing_data['label']
    
    # Initialize ensemble results
    ensemble_votes = np.zeros(len(y_test))
    ensemble_confidences = np.zeros(len(y_test))
    
    # Analyze individual models
    for model_name in models.keys():
        model = joblib.load(f'models/{model_name.lower().replace(" ", "_")}.joblib')
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Update ensemble votes and confidences
        if y_proba is not None:
            ensemble_votes += (y_proba > 0.5).astype(int)
            ensemble_confidences += y_proba
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Get classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[model_name] = {
            'report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_proba
        }
    
    # Calculate ensemble results
    ensemble_votes = ensemble_votes / len(models)  # Normalize votes
    ensemble_confidences = ensemble_confidences / len(models)  # Average confidence
    ensemble_predictions = (ensemble_votes > 0.5).astype(int)
    
    # Calculate ensemble metrics
    ensemble_cm = confusion_matrix(y_test, ensemble_predictions)
    ensemble_report = classification_report(y_test, ensemble_predictions, output_dict=True)
    
    # Add ensemble results
    results['Ensemble'] = {
        'report': ensemble_report,
        'confusion_matrix': ensemble_cm,
        'predictions': ensemble_predictions,
        'probabilities': ensemble_confidences,
        'votes': ensemble_votes
    }
    
    return results 