import pandas as pd

from signals.signal_engine import generate


def _feature_frame():
    return pd.DataFrame(
        [
            {
                "Close": 100.0,
                "RSI": 55.0,
                "MACD": 1.0,
                "MACD_sig": 0.5,
                "MACD_hist": 0.2,
                "BB_upper": 110.0,
                "BB_lower": 90.0,
                "BB_pct": 0.55,
                "SMA7": 102.0,
                "SMA30": 98.0,
                "Stoch_K": 55.0,
                "Stoch_D": 50.0,
                "Vol7": 1.0,
                "Vol30": 1.0,
            }
        ]
    )


def test_generate_confidence_stays_stable_under_high_volatility():
    feat = _feature_frame()
    forecast = pd.DataFrame({"Forecast": [101.0, 102.0, 103.0]})

    low_vol = generate(
        feat,
        forecast_df=forecast,
        sentiment=0.0,
        garch_info={"regime": "Low volatility"},
        ml_info={"prediction": "Up", "probability": 0.55},
        rl_info={"action": "HOLD", "confidence": 50.0},
    )
    high_vol = generate(
        feat,
        forecast_df=forecast,
        sentiment=0.0,
        garch_info={"regime": "High volatility"},
        ml_info={"prediction": "Up", "probability": 0.55},
        rl_info={"action": "HOLD", "confidence": 50.0},
    )

    assert low_vol["signal"] == "BUY"
    assert high_vol["signal"] == "BUY"
    assert low_vol["confidence"] >= 70
    assert high_vol["confidence"] >= 60
    assert low_vol["confidence"] - high_vol["confidence"] <= 15