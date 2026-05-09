"""Common training wrappers with early stopping and logging.

Provides small helper functions to train classifiers (XGBoost or sklearn
fallback) with optional early stopping and to train LSTM models with
callbacks. These wrappers return fitted models and any scalers used.
"""
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


def train_classifier_with_early_stopping(X_train: pd.DataFrame, y_train: pd.Series,
                                        X_val: pd.DataFrame, y_val: pd.Series,
                                        use_tuning: bool = False,
                                        params: Optional[dict] = None,
                                        early_stopping_rounds: int = 10) -> Tuple[object, object]:
    """Train a tree classifier with scaler and (for XGBoost) early stopping.

    Returns (fitted_model, fitted_scaler).
    """
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xval = scaler.transform(X_val)

    if XGBOOST_AVAILABLE:
        if params is None:
            params = {
                "n_estimators": 120,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": 42,
                "use_label_encoder": False,
                "verbosity": 0,
                "n_jobs": -1,
            }
        model = XGBClassifier(**params)
        # early stopping with eval_set
        try:
            model.fit(Xtr, y_train, eval_set=[(Xval, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=False)
        except TypeError:
            # older xgboost versions may not accept verbose keyword
            model.fit(Xtr, y_train, eval_set=[(Xval, y_val)], early_stopping_rounds=early_stopping_rounds)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        if params is None:
            params = {"max_iter": 120, "learning_rate": 0.06, "max_depth": 4, "random_state": 42}
        model = HistGradientBoostingClassifier(**params)
        model.fit(Xtr, y_train)

    return model, scaler


def train_lstm_with_wrapper(Xtr: np.ndarray, ytr: np.ndarray, Xval: Optional[np.ndarray], yval: Optional[np.ndarray],
                            units: int = 32, dropout: float = 0.2, epochs: int = 50, batch_size: int = 32, patience: int = 5):
    """Train a simple LSTM with early stopping. Expects reshaped inputs: (n, lookback, 1).

    Returns the fitted Keras model.
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except Exception:
        raise ImportError("TensorFlow not available — install tensorflow to train LSTM")

    import tensorflow as tf
    tf.keras.backend.clear_session()

    model = Sequential([
        LSTM(units, input_shape=(Xtr.shape[1], Xtr.shape[2])),
        Dropout(dropout),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    callbacks = [EarlyStopping(patience=patience, restore_best_weights=True)]
    if Xval is not None and len(Xval) > 0:
        model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks)
    else:
        model.fit(Xtr, ytr, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks)

    return model


__all__ = ["train_classifier_with_early_stopping", "train_lstm_with_wrapper"]
