"""Entain Sports Betting Platform — Domain Models."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4


# ═══════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════

class MatchEventType(str, Enum):
    """Types of events that occur during a match."""
    GOAL_SCORED = "goal_scored"
    RED_CARD = "red_card"
    YELLOW_CARD = "yellow_card"
    INJURY = "injury"
    SUBSTITUTION = "substitution"
    HALF_TIME = "half_time"
    FULL_TIME = "full_time"
    VAR_DECISION = "var_decision"
    PENALTY_AWARDED = "penalty_awarded"


class BetStatus(str, Enum):
    """Bet lifecycle states."""
    PENDING = "pending"
    ACTIVE = "active"
    WON = "won"
    LOST = "lost"
    VOID = "void"
    SETTLED = "settled"
    CASHED_OUT = "cashed_out"


class MarketStatus(str, Enum):
    """Market trading states."""
    OPEN = "open"
    SUSPENDED = "suspended"
    CLOSED = "closed"
    SETTLED = "settled"


# ═══════════════════════════════════════════════════════════════
# DOMAIN MODELS
# ═══════════════════════════════════════════════════════════════

@dataclass
class MatchEvent:
    """An event occurring during a live match."""
    id: UUID = field(default_factory=uuid4)
    match_id: str = ""
    event_type: MatchEventType = MatchEventType.GOAL_SCORED
    team: str = ""  # "home" or "away"
    player_name: str = ""
    minute: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


@dataclass
class BettingOdds:
    """Represents calculated odds for a market."""
    id: UUID = field(default_factory=uuid4)
    match_id: str = ""
    market_type: str = ""  # e.g., "match_result", "over_under_2.5"
    home_win: float = 0.0
    draw: float = 0.0
    away_win: float = 0.0
    margin_applied: float = 0.0
    calculated_at: datetime = field(default_factory=datetime.utcnow)
    previous_odds_id: UUID | None = None


@dataclass
class Bet:
    """Represents a customer's bet."""
    id: UUID = field(default_factory=uuid4)
    customer_id: UUID = field(default_factory=uuid4)
    match_id: str = ""
    selection: str = ""  # e.g., "home_win", "over_2.5"
    odds_at_placement: float = 1.0
    stake: float = 0.0
    potential_payout: float = 0.0
    status: BetStatus = BetStatus.PENDING
    placed_at: datetime = field(default_factory=datetime.utcnow)
    settled_at: datetime | None = None
    settlement_event_id: UUID | None = None


@dataclass
class Settlement:
    """Represents a bet settlement record."""
    id: UUID = field(default_factory=uuid4)
    bet_id: UUID = field(default_factory=uuid4)
    match_id: str = ""
    result: str = ""  # e.g., "home_win"
    payout_amount: float = 0.0
    settled_at: datetime = field(default_factory=datetime.utcnow)
    event_id: UUID = field(default_factory=uuid4)
    is_duplicate: bool = False


@dataclass
class Customer:
    """Represents a betting customer."""
    id: UUID = field(default_factory=uuid4)
    username: str = ""
    email: str = ""
    balance: float = 0.0
    is_verified: bool = False
    risk_level: str = "standard"  # standard, high, vip
    created_at: datetime = field(default_factory=datetime.utcnow)
