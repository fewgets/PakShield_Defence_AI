import pandas as pd
import joblib
import numpy as np
from io import StringIO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def predict_from_csv(csv_data, model_dir: Path | None = None):
    """
    Loads model, scaler, encoder from models folder,
    processes CSV data, and returns predicted labels.
    
    Args:
        csv_data: Can be a file path (str), pandas DataFrame, or file-like object
        model_dir: Directory containing the trained models
    
    Returns:
        Array of predicted labels
    """

    model_directory = model_dir or (BASE_DIR / "Models" / "IDS")

    model_path = Path(model_directory) / "ids_model.pkl"
    scaler_path = Path(model_directory) / "scaler.pkl"
    encoder_path = Path(model_directory) / "label_encoder.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Label encoder file not found at {encoder_path}")

    # Load saved objects
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)

    # Handle different input types
    if isinstance(csv_data, pd.DataFrame):
        df = csv_data.copy()
    elif isinstance(csv_data, str):
        # Assume it's a file path
        df = pd.read_csv(csv_data)
    else:
        # Assume it's a file-like object
        df = pd.read_csv(csv_data)

    # Drop unwanted columns
    drop_cols = ['Timestamp', 'Dst Port', 'Protocol']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Drop label if present (for prediction only)
    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    # Replace inf/-inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with 0 (or better: df.fillna(df.mean()))
    df = df.fillna(0)

    # Scale features
    X_scaled = scaler.transform(df)

    # Predict
    preds_num = model.predict(X_scaled)

    # Convert numeric preds back to labels
    preds = encoder.inverse_transform(preds_num)

    return preds

if __name__ == "__main__":
    # Example usage
    sample_csv = BASE_DIR / "Datasets" / "df_sample_500.csv"
    TEST = predict_from_csv(sample_csv)
    print(TEST)