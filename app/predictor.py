import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple
import os
from dotenv import load_dotenv

load_dotenv()


class ChurnPredictor:
    def __init__(self, model_path: str = None, encoders_path: str = None):
        self.model = None
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_order = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"
        ]
        self.categorical_columns = [
            "gender", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaymentMethod"
        ]

        if model_path is None:
            model_path = os.getenv("MODEL_PATH", "models/churn_model.pkl")
        if encoders_path is None:
            encoders_path = os.getenv("ENCODERS_PATH", "models/churn_encoders.pkl")

        self.load_model(model_path, encoders_path)

    def load_model(self, model_path: str, encoders_path: str):
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the model file exists.")

        try:
            self.encoders = joblib.load(encoders_path)
        except FileNotFoundError:
            print(f"Warning: Encoders file not found at {encoders_path}. Initializing new encoders.")
            self._initialize_encoders()

    def _initialize_encoders(self):
        for col in self.categorical_columns:
            self.encoders[col] = LabelEncoder()

    def preprocess(self, data: Dict) -> np.ndarray:
        df = pd.DataFrame([data])

        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

        for col in self.categorical_columns:
            if col in df.columns:
                if col in self.encoders and hasattr(self.encoders[col], "classes_"):
                    try:
                        df[col] = self.encoders[col].transform(df[col])
                    except ValueError:
                        df[col] = self.encoders[col].fit_transform(df[col])
                else:
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col])
                    self.encoders[col] = encoder

        features = df[self.feature_order].values[0]
        return features.reshape(1, -1)

    def predict(self, data: Dict) -> Tuple[int, float]:
        features = self.preprocess(data)
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0, 1]
        return int(prediction), float(probability)

    def get_churn_risk(self, probability: float) -> str:
        if probability < 0.3:
            return "Low"
        elif probability < 0.6:
            return "Medium"
        else:
            return "High"


predictor = ChurnPredictor()
