"""Entain Sports Betting Platform — Custom Exception Classes."""

from uuid import UUID


class EntainBaseError(Exception):
    """Base exception for all Entain application errors."""
    pass


# ═══════════════════════════════════════════════════════════════
# ODDS & TRADING EXCEPTIONS
# ═══════════════════════════════════════════════════════════════

class OddsCalculationError(EntainBaseError):
    """Raised when odds calculation produces invalid results."""
    def __init__(self, match_id: str, reason: str):
        self.match_id = match_id
        self.reason = reason
        super().__init__(f"Odds calculation failed for match {match_id}: {reason}")


class AnomalousOddsError(EntainBaseError):
    """Raised when calculated odds are outside acceptable range (< 1.01 or > 1000)."""
    def __init__(self, match_id: str, odds_value: float):
        self.match_id = match_id
        self.odds_value = odds_value
        super().__init__(f"Anomalous odds {odds_value} for match {match_id}")


class MarketSuspendedError(EntainBaseError):
    """Raised when attempting to publish odds to a suspended market."""
    def __init__(self, match_id: str, market_type: str):
        self.match_id = match_id
        self.market_type = market_type
        super().__init__(f"Market {market_type} suspended for match {match_id}")


class SuspiciousActivityError(EntainBaseError):
    """Raised when odds movement exceeds threshold (UKGC reportable)."""
    def __init__(self, match_id: str, movement_pct: float, window_seconds: int):
        self.match_id = match_id
        self.movement_pct = movement_pct
        self.window_seconds = window_seconds
        super().__init__(
            f"Suspicious odds movement for match {match_id}: "
            f"{movement_pct:.1%} in {window_seconds}s"
        )


# ═══════════════════════════════════════════════════════════════
# SETTLEMENT EXCEPTIONS
# ═══════════════════════════════════════════════════════════════

class BetNotFoundError(EntainBaseError):
    """Raised when a bet lookup returns no results."""
    def __init__(self, bet_id: UUID):
        self.bet_id = bet_id
        super().__init__(f"Bet not found: {bet_id}")


class DuplicateSettlementError(EntainBaseError):
    """Raised when a bet has already been settled (idempotency check)."""
    def __init__(self, bet_id: UUID, original_event_id: UUID):
        self.bet_id = bet_id
        self.original_event_id = original_event_id
        super().__init__(f"Bet {bet_id} already settled by event {original_event_id}")


class LockAcquisitionError(EntainBaseError):
    """Raised when distributed lock cannot be acquired after retries."""
    def __init__(self, bet_id: UUID, attempts: int):
        self.bet_id = bet_id
        self.attempts = attempts
        super().__init__(f"Failed to acquire lock for bet {bet_id} after {attempts} attempts")


class InsufficientBalanceError(EntainBaseError):
    """Raised when customer balance is insufficient for a payout."""
    def __init__(self, customer_id: UUID, required: float, available: float):
        self.customer_id = customer_id
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient platform balance for customer {customer_id}: "
            f"payout={required}, pool={available}"
        )
