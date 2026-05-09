"""
TradeMind AI — Alert System
==============================
Pluggable alert backend:
  - Console (default / dev)
  - Telegram Bot
  - Email (SMTP)

Add Telegram/Email credentials via environment variables:
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
  ALERT_EMAIL_FROM
  ALERT_EMAIL_TO
  ALERT_SMTP_HOST
  ALERT_SMTP_PORT
  ALERT_SMTP_PASSWORD
"""

import os
import smtplib
from abc import ABC, abstractmethod
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from datetime import datetime


# ─────────────────────────────────────────────
# BASE ALERT BACKEND
# ─────────────────────────────────────────────
class AlertBackend(ABC):
    @abstractmethod
    def send(self, title: str, message: str) -> bool:
        """Send an alert. Returns True on success."""
        ...


# ─────────────────────────────────────────────
# CONSOLE BACKEND (default)
# ─────────────────────────────────────────────
class ConsoleAlertBackend(AlertBackend):
    def send(self, title: str, message: str) -> bool:
        print(f"\n[ALERT] {title}\n{message}\n")
        return True


# ─────────────────────────────────────────────
# TELEGRAM BACKEND
# ─────────────────────────────────────────────
class TelegramAlertBackend(AlertBackend):
    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")

    def send(self, title: str, message: str) -> bool:
        if not self.token or not self.chat_id:
            print("[Alert] Telegram credentials not set.")
            return False
        try:
            import requests
            text = f"*{title}*\n\n{message}"
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            resp = requests.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown",
            }, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            print(f"[Alert] Telegram error: {e}")
            return False


# ─────────────────────────────────────────────
# EMAIL BACKEND
# ─────────────────────────────────────────────
class EmailAlertBackend(AlertBackend):
    def __init__(self):
        self.from_addr = os.getenv("ALERT_EMAIL_FROM", "")
        self.to_addr = os.getenv("ALERT_EMAIL_TO", "")
        self.smtp_host = os.getenv("ALERT_SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("ALERT_SMTP_PORT", "587"))
        self.password = os.getenv("ALERT_SMTP_PASSWORD", "")

    def send(self, title: str, message: str) -> bool:
        if not all([self.from_addr, self.to_addr, self.password]):
            print("[Alert] Email credentials not set.")
            return False
        try:
            msg = MIMEMultipart()
            msg["From"] = self.from_addr
            msg["To"] = self.to_addr
            msg["Subject"] = f"TradeMind AI Alert: {title}"
            msg.attach(MIMEText(message, "plain"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.from_addr, self.password)
                server.sendmail(self.from_addr, self.to_addr, msg.as_string())
            return True
        except Exception as e:
            print(f"[Alert] Email error: {e}")
            return False


# ─────────────────────────────────────────────
# ALERT MANAGER
# ─────────────────────────────────────────────
class AlertManager:
    """
    Send an alert through all registered backends.
    Usage:
        manager = AlertManager()
        manager.add_backend(TelegramAlertBackend())
        manager.signal_alert("BTC-USD", "BUY", 82, ["RSI oversold"])
    """

    def __init__(self):
        self.backends: list[AlertBackend] = [ConsoleAlertBackend()]

    def add_backend(self, backend: AlertBackend):
        self.backends.append(backend)

    def send(self, title: str, message: str):
        for backend in self.backends:
            backend.send(title, message)

    def signal_alert(self, ticker: str, signal: str, confidence: int, reasons: list):
        emoji_map = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
        emoji = emoji_map.get(signal, "⬜")
        title = f"{emoji} {signal} Signal — {ticker}"
        reason_text = "\n".join(f"• {r}" for r in reasons)
        message = (
            f"Ticker: {ticker}\n"
            f"Signal: {signal} {emoji}\n"
            f"Confidence: {confidence}%\n\n"
            f"Reasons:\n{reason_text}\n\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"— TradeMind AI"
        )
        self.send(title, message)
