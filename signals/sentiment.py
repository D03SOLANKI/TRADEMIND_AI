"""TradeMind AI — Sentiment Analysis"""
import pandas as pd
from typing import List


def analyze(text: str) -> dict:
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        p    = blob.sentiment.polarity
        s    = blob.sentiment.subjectivity
    except Exception:
        p, s = 0.0, 0.0

    if p > 0.1:   label, emoji = "Positive", "😀"
    elif p < -0.1: label, emoji = "Negative", "😞"
    else:          label, emoji = "Neutral",  "😐"

    return {"polarity": round(p,4), "subjectivity": round(s,4),
            "label": label, "emoji": emoji, "confidence": int(abs(p)*100)}


MOCK_HEADLINES = [
    "Markets rally as tech stocks surge on strong earnings results.",
    "Crypto assets recover amid renewed institutional buying interest.",
    "Federal Reserve signals potential pause in rate hike cycle.",
    "Recession fears persist as PMI data disappoints economists.",
    "Oil prices fall sharply on demand concerns from China slowdown.",
]


def analyze_bulk(texts: List[str]) -> pd.DataFrame:
    return pd.DataFrame([{**analyze(t), "text": t} for t in texts])[
        ["text","label","polarity","subjectivity","confidence","emoji"]
    ]


def _normalize_query(query: str) -> str:
    q = query.strip()
    q = q.replace("=X", "").replace(".NS", "").replace(".BO", "")
    q = q.replace("-USD", "").replace("/USD", "")
    q = q.replace("-", " ").replace("_", " ")
    return q.strip() or "Markets"


def _category_from_query(query: str) -> str:
    q = query.strip().lower()
    if not q:
        return "markets"
    if any(term in q for term in ["crypto", "bitcoin", "btc", "ethereum", "eth",
                                 "doge", "xrp", "ada", "sol", "bnb"]):
        return "crypto"
    if any(term in q for term in ["=x", "/", "usd", "eur", "gbp", "jpy", "inr",
                                 "aud", "cad", "chf", "nzd", "mxn", "sgd", "zar"]):
        if ".ns" not in q and ".bo" not in q and "crypto" not in q and any(sym in q for sym in ["=x", "/"]):
            return "forex"
        if any(term in q for term in ["forex", "fx", "currency"]):
            return "forex"
    if any(term in q for term in ["ns", "bo", "nifty", "sensex", "reliance", "tcs",
                                 "infy", "hdfc", "icici", "bharat", "adani", "tit", "itc"]):
        return "indian"
    if any(term in q for term in ["stock", "stocks", "equity", "nasdaq", "sp500", "dow",
                                 "apple", "aapl", "msft", "googl", "amzn", "nvda", "meta", "tsla"]):
        return "us stock"
    # Fallback keywords for tickers that look like US stock symbols.
    if q.isalpha() and 1 < len(q) <= 5:
        return "us stock"
    return "markets"


def _topic_headlines(category: str, query: str) -> List[str]:
    topic = _normalize_query(query)
    if category == "crypto":
        return [
            f"{topic} continues to attract buyers as crypto markets rally on renewed optimism.",
            f"Analysts say {topic} momentum is strengthening after positive network updates.",
            f"{topic} adoption news lifts the broader crypto sector and investor sentiment.",
            f"Whales accumulate {topic} while volatility stays subdued ahead of major event.",
            f"Trading volume around {topic} increases as market confidence improves.",
        ]
    if category == "us stock":
        return [
            f"{topic} shares rise as the US tech sector rebounds on strong earnings.",
            f"Investors express confidence in {topic} after favorable analyst revisions.",
            f"Market breadth improves with {topic} leading gains on Wall Street.",
            f"{topic} outlook brightens amid broader US economic recovery signs.",
            f"Fund flows into {topic} pickup as demand for growth stocks resumes.",
        ]
    if category == "indian":
        return [
            f"{topic} climbs on strong domestic demand and bullish India market sentiment.",
            f"Analysts remain upbeat on {topic} after positive earnings and macro updates.",
            f"India equities, including {topic}, benefit from stable rupee and inflows.",
            f"{topic} outlook is supported by favorable policy signals and consumption strength.",
            f"Investors favor {topic} amid rising optimism for Indian economic growth.",
        ]
    if category == "forex":
        return [
            f"{topic} moves on macro data and central bank commentary in forex markets.",
            f"Currency traders are watching {topic} after volatile global risk sentiment.",
            f"{topic} stabilizes as safe-haven demand shifts and liquidity improves.",
            f"Technical flows drive momentum around {topic} ahead of economic releases.",
            f"{topic} sentiment stays cautious as traders digest currency pair fundamentals.",
        ]
    return [
        f"Global markets react to fresh headline news around {topic}.",
        f"Sentiment for {topic} is mixed as investors weigh economic and geopolitical factors.",
        f"Analysts debate the outlook for {topic} amid shifting market momentum.",
        f"Headline interest in {topic} grows as investors seek clarity on the outlook.",
        f"Market pulse for {topic} remains steady with both bullish and bearish voices.",
    ]


def news_sentiment(query: str) -> dict:
    category = _category_from_query(query)
    headlines = _topic_headlines(category, query)
    results   = analyze_bulk(headlines)
    avg       = results["polarity"].mean()
    if avg > 0.05:   label, emoji = "Bullish", "📈"
    elif avg < -0.05: label, emoji = "Bearish", "📉"
    else:             label, emoji = "Neutral", "➡️"
    return {"polarity": round(avg,4), "label": label, "emoji": emoji,
            "n": len(headlines), "df": results}
