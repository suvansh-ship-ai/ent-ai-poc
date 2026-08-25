"""Entain Sports Betting Platform — Live Odds Calculation Service.

Implements SCRUM-1: recalculates in-play football odds in response to
match events (goals, cards, injuries, substitutions, etc.), applies the
platform margin, validates results, flags UKGC-reportable suspicious
movements, persists an immutable audit trail, and publishes updated
odds to the trading platform.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import structlog

from .config import config
from .exceptions import (
    AnomalousOddsError,
    MarketSuspendedError,
    OddsCalculationError,
    SuspiciousActivityError,
)
from .models import BettingOdds, MarketStatus, MatchEvent, MatchEventType

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════
# DEPENDENCY INTERFACES (Repository / Client Protocols)
# ═══════════════════════════════════════════════════════════════

class OddsRepository(Protocol):
    """Data-access contract for persisted odds records (append-only)."""

    async def get_latest_odds(self, match_id: str, market_type: str) -> BettingOdds | None:
        """Return the most recently calculated odds for a match/market, if any."""
        ...

    async def save_odds(self, odds: BettingOdds) -> BettingOdds:
        """Persist a new odds record. Odds history must never be updated/deleted."""
        ...

    async def get_odds_history(
        self, match_id: str, market_type: str, window_seconds: int
    ) -> list[BettingOdds]:
        """Return odds records calculated within the trailing time window, oldest first."""
        ...


class TradingPlatformClient(Protocol):
    """Contract for publishing odds to the downstream trading platform."""

    async def publish_odds(self, odds: BettingOdds) -> None:
        """Publish an odds update to the trading platform."""
        ...

    async def get_market_status(self, match_id: str, market_type: str) -> MarketStatus:
        """Return the current trading status of a market."""
        ...


# ═══════════════════════════════════════════════════════════════
# EVENT IMPACT MODEL
# ═══════════════════════════════════════════════════════════════

# Multiplicative adjustment applied to decimal odds for the event's
# (favoured team, opposing team, draw) legs. Values < 1.0 shorten odds
# (more likely outcome); values > 1.0 lengthen odds (less likely outcome).
_EventImpact = tuple[float, float, float]

_EVENT_IMPACT_FACTORS: dict[MatchEventType, _EventImpact] = {
    MatchEventType.GOAL_SCORED: (0.55, 1.35, 1.30),
    MatchEventType.PENALTY_AWARDED: (0.80, 1.10, 1.05),
    MatchEventType.RED_CARD: (1.45, 0.70, 1.10),  # "favoured" leg = the penalised team
    MatchEventType.YELLOW_CARD: (1.02, 0.99, 1.00),
    MatchEventType.INJURY: (1.05, 0.98, 1.01),
    MatchEventType.SUBSTITUTION: (1.00, 1.00, 1.00),
    MatchEventType.VAR_DECISION: (1.00, 1.00, 1.00),
    MatchEventType.HALF_TIME: (1.00, 1.00, 1.00),
    MatchEventType.FULL_TIME: (1.00, 1.00, 1.00),
}

DEFAULT_MARKET_TYPE = "match_result"
SUSPICIOUS_WINDOW_SECONDS = 60


# ═══════════════════════════════════════════════════════════════
# SERVICE
# ═══════════════════════════════════════════════════════════════

class OddsCalculationService:
    """Recalculates and publishes live in-play odds from match events.

    Triggered by incoming match events from the event stream, this
    service recomputes market odds, applies the platform margin,
    validates the result, checks for UKGC-reportable suspicious
    movements, persists an immutable audit trail, and publishes the
    updated odds to the trading platform.

    The service is stateless; all state is delegated to the injected
    repository and trading platform client.
    """

    def __init__(
        self,
        odds_repository: OddsRepository,
        trading_platform_client: TradingPlatformClient,
    ) -> None:
        self._odds = odds_repository
        self._trading = trading_platform_client

    async def calculate_live_odds(
        self,
        event: MatchEvent,
        market_type: str = DEFAULT_MARKET_TYPE,
    ) -> BettingOdds:
        """Calculate and publish updated odds based on a match event.

        Args:
            event: The match event received from the event stream (goal,
                red/yellow card, injury, substitution, etc.).
            market_type: The market to recalculate (default: ``match_result``).

        Returns:
            The newly calculated, persisted, and published :class:`BettingOdds`.

        Raises:
            OddsCalculationError: If odds cannot be calculated due to missing
                baseline data or an unexpected internal failure.
            AnomalousOddsError: If calculated odds fall outside the
                configured acceptable range.
            MarketSuspendedError: If the target market is suspended, closed,
                or settled on the trading platform.
        """
        correlation_id = str(event.id)

        # UKGC requirement: log BEFORE the operation for crash reconstruction.
        logger.info(
            "odds_calculation_started",
            correlation_id=correlation_id,
            match_id=event.match_id,
            market_type=market_type,
            event_type=event.event_type.value,
            event_team=event.team,
            event_minute=event.minute,
            actor="system:odds-calculation-service",
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

        try:
            market_status = await self._trading.get_market_status(event.match_id, market_type)
            if market_status in (MarketStatus.SUSPENDED, MarketStatus.CLOSED, MarketStatus.SETTLED):
                logger.warning(
                    "odds_calculation_market_unavailable",
                    correlation_id=correlation_id,
                    match_id=event.match_id,
                    market_type=market_type,
                    market_status=market_status.value,
                )
                raise MarketSuspendedError(event.match_id, market_type)

            current_odds = await self._odds.get_latest_odds(event.match_id, market_type)
            if current_odds is None:
                raise OddsCalculationError(
                    event.match_id,
                    f"no baseline odds found for market '{market_type}'",
                )

            new_odds = self._apply_event_impact(current_odds, event, market_type)
            new_odds = self._apply_margin(new_odds)
            self._validate_odds(new_odds)

            await self._check_suspicious_movement(current_odds, new_odds, market_type, correlation_id)

            new_odds.previous_odds_id = current_odds.id
            new_odds.calculated_at = datetime.utcnow()

            saved_odds = await self._odds.save_odds(new_odds)
            await self._trading.publish_odds(saved_odds)

            # UKGC requirement: log AFTER the operation with before/after values.
            logger.info(
                "odds_calculation_completed",
                correlation_id=correlation_id,
                match_id=event.match_id,
                market_type=market_type,
                actor="system:odds-calculation-service",
                reason=f"match_event:{event.event_type.value}",
                before={
                    "home_win": current_odds.home_win,
                    "draw": current_odds.draw,
                    "away_win": current_odds.away_win,
                    "odds_id": str(current_odds.id),
                },
                after={
                    "home_win": saved_odds.home_win,
                    "draw": saved_odds.draw,
                    "away_win": saved_odds.away_win,
                    "odds_id": str(saved_odds.id),
                },
                timestamp=saved_odds.calculated_at.isoformat() + "Z",
            )

            return saved_odds

        except (OddsCalculationError, AnomalousOddsError, MarketSuspendedError, SuspiciousActivityError):
            logger.error(
                "odds_calculation_failed",
                correlation_id=correlation_id,
                match_id=event.match_id,
                market_type=market_type,
                event_type=event.event_type.value,
            )
            raise
        except Exception as exc:  # noqa: BLE001 — wrap all unexpected errors
            logger.error(
                "odds_calculation_unexpected_error",
                correlation_id=correlation_id,
                match_id=event.match_id,
                market_type=market_type,
                error=str(exc),
            )
            raise OddsCalculationError(event.match_id, f"unexpected error: {exc}") from exc

    def _apply_event_impact(
        self, current_odds: BettingOdds, event: MatchEvent, market_type: str
    ) -> BettingOdds:
        """Return new odds reflecting the impact of a match event.

        Args:
            current_odds: The last known odds for the market.
            event: The triggering match event.
            market_type: The market being recalculated.

        Returns:
            A new, unsaved :class:`BettingOdds` instance with adjusted prices
            (margin not yet reapplied).
        """
        factors = _EVENT_IMPACT_FACTORS.get(event.event_type)
        if factors is None:
            logger.warning(
                "odds_calculation_no_impact_model",
                match_id=event.match_id,
                event_type=event.event_type.value,
            )
            factors = (1.0, 1.0, 1.0)

        favoured_factor, opposing_factor, draw_factor = factors

        if event.team == "home":
            home_multiplier, away_multiplier = favoured_factor, opposing_factor
        elif event.team == "away":
            home_multiplier, away_multiplier = opposing_factor, favoured_factor
        else:
            home_multiplier, away_multiplier = 1.0, 1.0

        return BettingOdds(
            match_id=current_odds.match_id,
            market_type=market_type,
            home_win=current_odds.home_win * home_multiplier,
            draw=current_odds.draw * draw_factor,
            away_win=current_odds.away_win * away_multiplier,
            margin_applied=current_odds.margin_applied,
        )

    def _apply_margin(self, odds: BettingOdds) -> BettingOdds:
        """Normalise odds to fair probabilities and reapply the platform margin.

        Args:
            odds: Odds to adjust (raw, event-impacted prices).

        Returns:
            The same odds instance with margin-adjusted prices, rounded to 2dp.
        """
        implied_probabilities = [1 / odds.home_win, 1 / odds.draw, 1 / odds.away_win]
        total_probability = sum(implied_probabilities)
        fair_probabilities = [p / total_probability for p in implied_probabilities]

        target_overround = 1 + config.default_margin_pct
        margined_probabilities = [p * target_overround for p in fair_probabilities]

        odds.home_win = round(1 / margined_probabilities[0], 2)
        odds.draw = round(1 / margined_probabilities[1], 2)
        odds.away_win = round(1 / margined_probabilities[2], 2)
        odds.margin_applied = config.default_margin_pct
        return odds

    def _validate_odds(self, odds: BettingOdds) -> None:
        """Validate calculated odds are within the acceptable trading range.

        Args:
            odds: The calculated odds to validate.

        Raises:
            AnomalousOddsError: If any price is outside
                ``[config.min_odds_value, config.max_odds_value]``.
        """
        for value in (odds.home_win, odds.draw, odds.away_win):
            if not (config.min_odds_value <= value <= config.max_odds_value):
                raise AnomalousOddsError(odds.match_id, value)

    async def _check_suspicious_movement(
        self,
        previous_odds: BettingOdds,
        new_odds: BettingOdds,
        market_type: str,
        correlation_id: str,
    ) -> None:
        """Detect and log UKGC-reportable suspicious odds movements.

        Compares the new odds against the odds recorded at the start of the
        trailing ``SUSPICIOUS_WINDOW_SECONDS`` window. Movements exceeding
        ``config.suspicious_movement_threshold`` are logged as compliance
        alerts. This check is observational only and does NOT block
        publishing, since legitimate match events (e.g. goals) can
        legitimately cause large swings.

        Args:
            previous_odds: The odds immediately prior to this calculation.
            new_odds: The newly calculated odds.
            market_type: The market being recalculated.
            correlation_id: Correlation ID for structured log tracing.
        """
        history = await self._odds.get_odds_history(
            new_odds.match_id, market_type, SUSPICIOUS_WINDOW_SECONDS
        )
        window_start_odds = history[0] if history else previous_odds

        for label, before, after in (
            ("home_win", window_start_odds.home_win, new_odds.home_win),
            ("draw", window_start_odds.draw, new_odds.draw),
            ("away_win", window_start_odds.away_win, new_odds.away_win),
        ):
            if before <= 0:
                continue
            movement_pct = abs(after - before) / before
            if movement_pct > config.suspicious_movement_threshold:
                alert = SuspiciousActivityError(new_odds.match_id, movement_pct, SUSPICIOUS_WINDOW_SECONDS)
                logger.warning(
                    "suspicious_odds_movement_detected",
                    correlation_id=correlation_id,
                    match_id=new_odds.match_id,
                    market_type=market_type,
                    selection=label,
                    movement_pct=round(movement_pct, 4),
                    window_seconds=SUSPICIOUS_WINDOW_SECONDS,
                    before=before,
                    after=after,
                    alert=True,
                    ukgc_reportable=True,
                    detail=str(alert),
                )
