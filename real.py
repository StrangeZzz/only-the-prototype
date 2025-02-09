import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os

from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, RepeatVector, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set up logging for debugging and info messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set TensorFlow random seed for reproducibility
tf.keras.utils.set_random_seed(42)
np.random.seed(42)

# ---------------------------------------------
# Module 1: Monte Carlo Simulation for Risk Quantification
# ---------------------------------------------
def monte_carlo_simulation_vectorized(S0, mu, sigma, T, dt, iterations, use_antithetic=False):
    """
    Vectorized simulation of asset price paths using geometric Brownian motion.
    Optionally uses antithetic variates for variance reduction.
    
    Parameters:
        S0 (float): Initial asset price. Must be > 0.
        mu (float): Expected return (drift).
        sigma (float): Volatility. Must be > 0.
        T (float): Total time (years). Must be > 0.
        dt (float): Time step. Must be > 0 and < T.
        iterations (int): Number of simulated paths. Must be > 0.
        use_antithetic (bool): If True, iterations must be even.
        
    Returns:
        prices (ndarray): 2D array of shape (iterations, N) containing simulated price paths.
    """
    # Input validation
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if dt <= 0 or dt >= T:
        raise ValueError("dt must be positive and less than T.")
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if use_antithetic and iterations % 2 != 0:
        raise ValueError("iterations must be even when use_antithetic is True.")
    
    N = int(T / dt)
    n_paths = iterations // 2 if use_antithetic else iterations
    
    # Generate normal increments
    dW = np.random.normal(0, np.sqrt(dt), size=(n_paths, N-1))
    if use_antithetic:
        dW = np.vstack((dW, -dW))
    
    # Prepend zeros for t=0 and compute cumulative sum for Brownian motion
    dW = np.concatenate((np.zeros((dW.shape[0], 1)), dW), axis=1)
    W = np.cumsum(dW, axis=1)
    
    time = np.linspace(0, T, N)
    drift = (mu - 0.5 * sigma**2) * time  # shape (N,)
    diffusion = sigma * W  # shape (iterations, N)
    
    prices = S0 * np.exp(drift + diffusion)
    return prices

def plot_mc_simulation(prices, title="Monte Carlo Simulation: Sample Price Paths"):
    """Visualizes up to 50 sample paths from the Monte Carlo simulation."""
    plt.figure(figsize=(10, 6))
    for i in range(min(50, prices.shape[0])):
        plt.plot(prices[i, :], alpha=0.6)
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()

def test_monte_carlo_validation():
    """Tests invalid inputs for Monte Carlo simulation to ensure ValueErrors are raised."""
    try:
        monte_carlo_simulation_vectorized(-100, 0.05, 0.1, 1.0, 0.01, 1000)
    except ValueError as e:
        logging.info("Expected error for negative S0: %s", e)
    try:
        monte_carlo_simulation_vectorized(100, 0.05, -0.1, 1.0, 0.01, 1000)
    except ValueError as e:
        logging.info("Expected error for negative sigma: %s", e)
    try:
        monte_carlo_simulation_vectorized(100, 0.05, 0.1, -1.0, 0.01, 1000)
    except ValueError as e:
        logging.info("Expected error for negative T: %s", e)
    try:
        monte_carlo_simulation_vectorized(100, 0.05, 0.1, 1.0, 1.0, 1000)
    except ValueError as e:
        logging.info("Expected error for dt >= T: %s", e)
    try:
        monte_carlo_simulation_vectorized(100, 0.05, 0.1, 1.0, 0.01, 101, use_antithetic=True)
    except ValueError as e:
        logging.info("Expected error for odd iterations with antithetic=True: %s", e)

# ---------------------------------------------
# Module 2: Machine Learning for Risk Scoring and Loan Default Prediction
# ---------------------------------------------
def ml_risk_scoring_with_pipeline():
    """
    Creates a synthetic imbalanced dataset for loan defaults, builds a pipeline that
    includes StandardScaler, SMOTE, and a classifier. Uses GridSearchCV to tune
    hyperparameters for RandomForestClassifier, XGBClassifier, and LogisticRegression.
    Outputs classification reports.
    """
    X, y = make_classification(n_samples=2000, n_features=15, n_informative=8, 
                               n_redundant=3, weights=[0.75, 0.25],
                               random_state=42)
    
    logging.info("Original class distribution:\n%s", pd.Series(y).value_counts())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
    ])
    
    param_grid_rf = {
        'classifier': [RandomForestClassifier(class_weight='balanced', random_state=42)],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10]
    }
    
    # Compute scale_pos_weight for XGBoost based on training data
    scale_pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)
    param_grid_xgb = {
        'classifier': [XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                                       scale_pos_weight=scale_pos_weight, random_state=42)],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 6]
    }
    
    param_grid_lr = {
        'classifier': [LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)],
        'classifier__C': [0.1, 1.0, 10.0]
    }
    
    grids = {
        'Random Forest': param_grid_rf,
        'XGBoost': param_grid_xgb,
        'Logistic Regression': param_grid_lr
    }
    
    best_models = {}
    for name, grid in grids.items():
        logging.info("Tuning hyperparameters for %s...", name)
        grid_search = GridSearchCV(pipeline, grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        logging.info("%s best params: %s", name, grid_search.best_params_)
        y_pred = grid_search.predict(X_test)
        report = classification_report(y_test, y_pred)
        logging.info("\n--- %s Classification Report ---\n%s", name, report)
        print(f"\n--- {name} Classification Report ---\n{report}")
    
    return best_models

# ---------------------------------------------
# Module 3: Deep Learning for Market Trend Forecasting (LSTM)
# ---------------------------------------------
def dl_market_forecasting():
    """
    Uses a synthetic market price index time series, normalizes it, prepares it using sliding windows,
    and builds an LSTM model (with Bidirectional layers) to forecast future trends.
    Implements EarlyStopping and ModelCheckpoint, then reloads the saved model to verify predictions.
    Also plots actual vs. predicted values.
    """
    np.random.seed(42)
    t = np.arange(0, 300)
    data = 50 + 0.15 * t + 8 * np.sin(0.1 * t) + np.random.normal(scale=2, size=len(t))
    
    plt.figure(figsize=(10, 4))
    plt.plot(t, data, label='Raw Price Index')
    plt.title("Synthetic Market Price Index")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_norm = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    def create_time_series_dataset(series, window_size):
        X, y = [], []
        for i in range(len(series) - window_size):
            X.append(series[i:i+window_size])
            y.append(series[i+window_size])
        return np.array(X), np.array(y)
    
    window_size = 15
    X, y = create_time_series_dataset(data_norm, window_size)
    X = X.reshape((X.shape[0], window_size, 1))
    
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Use MeanSquaredError loss object to avoid serialization issues.
    lstm_model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(window_size, 1)),
        Dropout(0.2),
        Bidirectional(LSTM(64)),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint("best_lstm_model.keras", monitor='val_loss', save_best_only=True)
    ]
    
    history = lstm_model.fit(X_train, y_train, epochs=150, batch_size=16,
                             validation_split=0.2, callbacks=callbacks, verbose=0)
    
    # Reload best model to verify proper saving/loading
    if os.path.exists("best_lstm_model.keras"):
        best_model = load_model("best_lstm_model.keras", compile=True)
        logging.info("Successfully loaded best LSTM model from best_lstm_model.keras.")
    else:
        best_model = lstm_model
        logging.warning("Model checkpoint file not found; using current model.")
    
    y_pred = best_model.predict(X_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    plt.figure(figsize=(10, 4))
    plt.plot(y_test_inv, label="Actual")
    plt.plot(y_pred_inv, label="Predicted")
    plt.title("LSTM Market Trend Forecasting: Actual vs. Predicted")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    test_loss = best_model.evaluate(X_test, y_test, verbose=0)
    logging.info("LSTM Test Loss (MSE): %.4f", test_loss)

# ---------------------------------------------
# Module 4: Deep Learning for Anomaly Detection using an LSTM Autoencoder
# ---------------------------------------------
def dl_anomaly_detection_autoencoder():
    """
    Creates a synthetic time series with injected anomalies, normalizes it,
    builds an LSTM autoencoder model to reconstruct the time series,
    computes reconstruction errors (using MSE), and flags anomalies.
    Also plots the reconstruction error vs. threshold.
    """
    np.random.seed(42)
    t = np.arange(0, 350)
    data = 50 + 0.1 * t + 8 * np.sin(0.08 * t) + np.random.normal(scale=2, size=len(t))
    anomaly_idx = np.random.choice(len(data), size=7, replace=False)
    data[anomaly_idx] += np.random.choice([-25, 25], size=7)
    
    plt.figure(figsize=(10, 4))
    plt.plot(t, data, label="Time Series with Anomalies")
    plt.title("Synthetic Time Series with Injected Anomalies")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_norm = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    def create_ae_dataset(series, window_size):
        X = []
        for i in range(len(series) - window_size):
            X.append(series[i:i+window_size])
        return np.array(X)
    
    window_size = 20
    X_ae = create_ae_dataset(data_norm, window_size)
    X_ae = X_ae.reshape((X_ae.shape[0], window_size, 1))
    
    # Build LSTM Autoencoder
    input_layer = Input(shape=(window_size, 1))
    encoded = LSTM(32, activation='relu', return_sequences=False)(input_layer)
    bottleneck = Dense(16, activation='relu')(encoded)
    decoded = RepeatVector(window_size)(bottleneck)
    decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
    decoded = Dense(1)(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    
    autoencoder.fit(X_ae, X_ae, epochs=100, batch_size=16, validation_split=0.2,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                    verbose=0)
    
    X_recon = autoencoder.predict(X_ae)
    # Compute mean squared error per window
    recon_errors = np.mean(np.square(X_ae - X_recon), axis=(1, 2))
    
    # Set dynamic threshold for anomalies
    threshold = np.mean(recon_errors) + 2 * np.std(recon_errors)
    anomalies = recon_errors > threshold
    
    plt.figure(figsize=(10, 4))
    plt.plot(recon_errors, label="Reconstruction Error")
    plt.axhline(y=threshold, color="red", linestyle="--", label="Threshold")
    plt.title("LSTM Autoencoder Reconstruction Errors")
    plt.xlabel("Window Index")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    logging.info("Anomalous windows detected: %d", np.sum(anomalies))

# ---------------------------------------------
# Main Function: Integration of All Modules with Error Handling and Testing
# ---------------------------------------------
def main():
    try:
        # Test Monte Carlo input validations
        logging.info("Testing Monte Carlo input validations...")
        test_monte_carlo_validation()
        
        # Module 1: Monte Carlo Simulation
        logging.info("Running Monte Carlo Simulation for Risk Quantification...")
        S0, mu, sigma, T, dt, iterations = 100, 0.05, 0.1, 1.0, 0.01, 1000
        mc_paths = monte_carlo_simulation_vectorized(S0, mu, sigma, T, dt, iterations, use_antithetic=True)
        # Validate output shape
        expected_shape = (iterations, int(T/dt))
        assert mc_paths.shape == expected_shape, f"Expected shape {expected_shape}, got {mc_paths.shape}"
        plot_mc_simulation(mc_paths, title="Monte Carlo Simulation: 1000 Price Paths (Antithetic Variates)")
        
        # Plot distribution of final prices
        final_prices = mc_paths[:, -1]
        plt.figure(figsize=(8, 4))
        plt.hist(final_prices, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        plt.title("Distribution of Final Prices from Monte Carlo Simulation")
        plt.xlabel("Price")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()
        
        # Module 2: Machine Learning for Risk Scoring and Loan Default Prediction
        logging.info("Running ML Risk Scoring and Loan Default Prediction...")
        best_ml_models = ml_risk_scoring_with_pipeline()
        
        # Module 3: Deep Learning for Market Trend Forecasting
        logging.info("Running Market Trend Forecasting with LSTM...")
        dl_market_forecasting()
        
        # Module 4: Deep Learning for Anomaly Detection using LSTM Autoencoder
        logging.info("Running Anomaly Detection with LSTM Autoencoder...")
        dl_anomaly_detection_autoencoder()
    
    except KeyboardInterrupt:
        logging.info("Execution interrupted by user.")
    except Exception as e:
        logging.error("An error occurred: %s", e)

if __name__ == "__main__":
    main()
