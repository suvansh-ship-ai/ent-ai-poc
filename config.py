"""Entain Sports Betting Platform — Application Configuration."""

import os
from dataclasses import dataclass


@dataclass
class AppConfig:
    """Application configuration loaded from environment variables."""
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "postgresql://localhost:5432/entain_betting")
    
    # DynamoDB (for distributed locks)
    dynamodb_table_locks: str = os.getenv("DYNAMODB_LOCKS_TABLE", "entain-settlement-locks")
    dynamodb_region: str = os.getenv("AWS_REGION", "eu-west-1")
    lock_ttl_seconds: int = int(os.getenv("LOCK_TTL_SECONDS", "30"))
    
    # Event Stream (match events)
    event_stream_url: str = os.getenv("EVENT_STREAM_URL", "https://events.entain-internal.com")
    event_stream_api_key: str = os.getenv("EVENT_STREAM_API_KEY", "")
    
    # Trading Platform (odds publishing)
    trading_platform_url: str = os.getenv("TRADING_PLATFORM_URL", "https://trading.entain-internal.com")
    trading_platform_api_key: str = os.getenv("TRADING_PLATFORM_API_KEY", "")
    
    # Odds Configuration
    default_margin_pct: float = float(os.getenv("DEFAULT_MARGIN_PCT", "0.06"))  # 6% margin
    max_odds_value: float = 1000.0
    min_odds_value: float = 1.01
    suspicious_movement_threshold: float = 0.20  # 20% change in < 60s triggers alert
    
    # Settlement
    max_settlement_retries: int = int(os.getenv("MAX_SETTLEMENT_RETRIES", "3"))
    settlement_lock_ttl: int = int(os.getenv("SETTLEMENT_LOCK_TTL", "30"))
    
    # Application
    app_name: str = "entain-betting-platform"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # API
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))


# Global config instance
config = AppConfig()
