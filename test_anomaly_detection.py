import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def test_data_loading():
    df = pd.read_csv("simulated_health_data.csv")
    assert not df.empty
    assert set(['heart_rate', 'spo2', 'label']).issubset(df.columns)

def test_preprocessing():
    df = pd.read_csv("simulated_health_data.csv")
    X = df[["heart_rate", "spo2"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Check shape and mean/std
    assert X_scaled.shape == X.shape
    np.testing.assert_almost_equal(np.mean(X_scaled, axis=0), [0, 0], decimal=1)
    np.testing.assert_almost_equal(np.std(X_scaled, axis=0), [1, 1], decimal=1)

def test_model_inference():
    df = pd.read_csv("simulated_health_data.csv")
    X = df[["heart_rate", "spo2"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(n_estimators=10, contamination=0.02, random_state=42)
    preds = model.fit_predict(X_scaled)
    # Should only contain -1 and 1
    assert set(np.unique(preds)).issubset({-1, 1})
    assert len(preds) == len(df)

if __name__ == "__main__":
    test_data_loading()
    test_preprocessing()
    test_model_inference()
    print("All tests passed.")