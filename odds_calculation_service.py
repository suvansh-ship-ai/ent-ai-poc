"""Entain Sports Betting Platform — Live Odds Calculation Service.

Recalculates in-play "match_result" odds in response to live football match
events (goals, red cards, injuries, substitutions, etc.), publishes the
updated odds to the trading platform, and records a UKGC-compliant audit
trail of every calculation performed.
"""

from __future__ import annotations

from datetime import datetime
from typing import AsyncIterator, Protocol
from uuid import UUID

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

# The market this service currently calculates. Additional in-play markets
# (e.g. over/under, both-teams-to-score) can be added following the same
# pattern in future iterations.
MATCH_RESULT_MARKET: str = "match_result"


# ═══════════════════════════════════════════════════════════════
# DEPENDENCY CONTRACTS (repository / client interfaces)
# ═══════════════════════════════════════════════════════════════

class OddsRepository(Protocol):
    """Data access contract for BettingOdds persistence."""

    async def get_latest_odds(self, match_id: str, market_type: str) -> BettingOdds | None:
        """Return the most recently calculated odds for a match/market, if any."""
        ...

    async def save(self, odds: BettingOdds) -> BettingOdds:
        """Persist a newly calculated odds record."""
        ...

    async def get_market_status(self, match_id: str, market_type: str) -> MarketStatus:
        """Return the current trading status of a match/market."""
        ...


class TradingPlatformClient(Protocol):
    """Contract for publishing odds updates to the downstream trading platform."""

    async def publish_odds(self, odds: BettingOdds) -> None:
        """Publish a BettingOdds update so it becomes live for customers."""
        ...


class EventStreamClient(Protocol):
    """Contract for consuming live match events."""

    def subscribe(self) -> AsyncIterator[MatchEvent]:
        """Yield live MatchEvent instances as they occur."""
        ...


class AuditLogger(Protocol):
    """Contract for the UKGC-compliant, append-only odds audit trail."""

    async def record_odds_change(self, audit_entry: dict) -> None:
        """Persist an immutable audit entry describing an odds change."""
        ...


# ═══════════════════════════════════════════════════════════════
# EVENT IMPACT MODEL
# ═══════════════════════════════════════════════════════════════

# Relative strength of each event type's effect on match-result probabilities.
# Values are fractions of probability mass shifted between the two sides.
_EVENT_IMPACT_WEIGHT: dict[MatchEventType, float] = {
    MatchEventType.GOAL_SCORED: 0.14,
    MatchEventType.PENALTY_AWARDED: 0.08,
    MatchEventType.RED_CARD: 0.10,
    MatchEventType.VAR_DECISION: 0.03,
    MatchEventType.INJURY: 0.02,
    MatchEventType.YELLOW_CARD: 0.005,
    MatchEventType.SUBSTITUTION: 0.0,
    MatchEventType.HALF_TIME: 0.0,
    MatchEventType.FULL_TIME: 0.0,
}

# Event types where the impact FAVOURS the team recorded on the event
# (event.team). All other weighted event types disadvantage that team
# (e.g. a red card against the home team favours the away team).
_FAVOURS_ACTING_TEAM: frozenset[MatchEventType] = frozenset(
    {MatchEventType.GOAL_SCORED, MatchEventType.PENALTY_AWARDED}
)

_MIN_PROBABILITY: float = 0.01
_MAX_PROBABILITY: float = 0.97


class OddsCalculationService:
    """Calculates and publishes live in-play odds for football matches.

    Consumes match events (goals, red cards, injuries, substitutions, etc.),
    recalculates the affected market's odds, validates the result, and
    publishes it to the trading platform. Every calculation is recorded in
    an immutable audit trail to satisfy UKGC regulatory requirements.
    """

    def __init__(
        self,
        odds_repo: OddsRepository,
        event_stream: EventStreamClient,
        trading_platform: TradingPlatformClient,
        audit_logger: AuditLogger,
    ) -> None:
        """Initialise the service with its collaborators.

        Args:
            odds_repo: Repository used to read/write BettingOdds records.
            event_stream: Client used to subscribe to live match events.
            trading_platform: Client used to publish odds updates.
            audit_logger: Append-only audit trail writer (UKGC requirement).
        """
        self._odds = odds_repo
        self._events = event_stream
        self._trading = trading_platform
        self._audit = audit_logger

    async def run(self) -> None:
        """Continuously consume match events and recalculate odds.

        A failure processing a single event is logged and does not stop the
        consumer loop, so a problem with one match never halts odds updates
        for other live matches.
        """
        log = logger.bind(component="odds_calculation_service")
        log.info("event_stream_consumption_started")

        async for event in self._events.subscribe():
            try:
                await self.calculate_live_odds(event)
            except MarketSuspendedError as exc:
                log.warning("skipped_suspended_market", match_id=event.match_id, error=str(exc))
            except SuspiciousActivityError as exc:
                log.warning(
                    "skipped_suspicious_odds_movement", match_id=event.match_id, error=str(exc)
                )
            except OddsCalculationError as exc:
                log.error("odds_calculation_error", match_id=event.match_id, error=str(exc))
            except Exception as exc:  # noqa: BLE001 - never crash the consumer loop
                log.error(
                    "unexpected_error_processing_event", match_id=event.match_id, error=str(exc)
                )

    async def calculate_live_odds(self, event: MatchEvent) -> BettingOdds:
        """Recalculate and publish live match-result odds for a match event.

        Args:
            event: The live match event (goal, red card, injury, substitution, etc.)
                that triggers recalculation.

        Returns:
            The newly calculated and published BettingOdds.

        Raises:
            MarketSuspendedError: If the market is suspended or closed for trading.
            AnomalousOddsError: If the calculated odds fall outside acceptable bounds.
            SuspiciousActivityError: If the odds movement exceeds the UKGC-reportable
                threshold within the monitoring window.
            OddsCalculationError: If calculation, persistence or publishing fails.
        """
        log = logger.bind(correlation_id=str(event.id), match_id=event.match_id)
        log.info(
            "odds_calculation_started",
            event_type=event.event_type.value,
            team=event.team,
            minute=event.minute,
        )

        market_status = await self._get_market_status(event.match_id, log)
        if market_status in (MarketStatus.SUSPENDED, MarketStatus.CLOSED):
            log.warning("market_not_tradeable", market_status=market_status.value)
            raise MarketSuspendedError(event.match_id, MATCH_RESULT_MARKET)

        previous_odds = await self._odds.get_latest_odds(event.match_id, MATCH_RESULT_MARKET)

        # UKGC requirement: log BEFORE the operation so the change can be
        # reconstructed even if the process crashes mid-calculation.
        await self._audit.record_odds_change(
            self._build_audit_entry(
                event=event,
                phase="before",
                before=previous_odds,
                after=None,
            )
        )

        new_odds = self._compute_new_odds(event, previous_odds)
        self._validate_odds_bounds(new_odds)

        if previous_odds is not None:
            self._check_suspicious_movement(previous_odds, new_odds)

        try:
            saved_odds = await self._odds.save(new_odds)
        except Exception as exc:
            log.error("odds_persistence_failed", error=str(exc))
            raise OddsCalculationError(event.match_id, f"failed to persist odds: {exc}") from exc

        # UKGC requirement: log AFTER the operation with before/after values.
        await self._audit.record_odds_change(
            self._build_audit_entry(
                event=event,
                phase="after",
                before=previous_odds,
                after=saved_odds,
            )
        )

        try:
            await self._trading.publish_odds(saved_odds)
        except Exception as exc:
            log.error("odds_publish_failed", error=str(exc))
            raise OddsCalculationError(
                event.match_id, f"failed to publish odds to trading platform: {exc}"
            ) from exc

        log.info(
            "odds_calculation_completed",
            market_type=MATCH_RESULT_MARKET,
            home_win=saved_odds.home_win,
            draw=saved_odds.draw,
            away_win=saved_odds.away_win,
            margin_applied=saved_odds.margin_applied,
        )

        return saved_odds

    async def _get_market_status(self, match_id: str, log: structlog.BoundLogger) -> MarketStatus:
        """Fetch the current market status, wrapping unexpected errors."""
        try:
            return await self._odds.get_market_status(match_id, MATCH_RESULT_MARKET)
        except Exception as exc:
            log.error("market_status_lookup_failed", error=str(exc))
            raise OddsCalculationError(match_id, f"failed to check market status: {exc}") from exc

    def _compute_new_odds(self, event: MatchEvent, previous: BettingOdds | None) -> BettingOdds:
        """Derive new match-result odds from the previous odds and the triggering event.

        Args:
            event: The triggering match event.
            previous: The previously published odds, or None if this is the first
                calculation for the match (in which case even probabilities are used).

        Returns:
            A new, unsaved BettingOdds instance reflecting the event's impact.
        """
        home_prob, draw_prob, away_prob = self._current_fair_probabilities(previous)
        home_delta, away_delta = self._directional_impact(event)

        home_prob += home_delta
        away_prob += away_delta
        draw_prob -= home_delta + away_delta

        home_prob, draw_prob, away_prob = self._normalise_probabilities(
            home_prob, draw_prob, away_prob
        )

        margin = config.default_margin_pct
        overround = 1.0 + margin

        return BettingOdds(
            match_id=event.match_id,
            market_type=MATCH_RESULT_MARKET,
            home_win=round(overround / home_prob, 2),
            draw=round(overround / draw_prob, 2),
            away_win=round(overround / away_prob, 2),
            margin_applied=margin,
            previous_odds_id=previous.id if previous else None,
        )

    @staticmethod
    def _current_fair_probabilities(previous: BettingOdds | None) -> tuple[float, float, float]:
        """De-vig the previous odds to obtain fair (margin-free) probabilities.

        Returns even (1/3, 1/3, 1/3) probabilities if there is no prior odds record.
        """
        if previous is None:
            return (1 / 3, 1 / 3, 1 / 3)

        raw_home = 1 / previous.home_win
        raw_draw = 1 / previous.draw
        raw_away = 1 / previous.away_win
        total = raw_home + raw_draw + raw_away

        return (raw_home / total, raw_draw / total, raw_away / total)

    @staticmethod
    def _directional_impact(event: MatchEvent) -> tuple[float, float]:
        """Determine the (home_delta, away_delta) probability shift for an event.

        A positive delta increases that side's win probability; a negative delta
        decreases it. The draw probability absorbs the inverse of the combined shift.
        """
        weight = _EVENT_IMPACT_WEIGHT.get(event.event_type, 0.0)
        if weight == 0.0 or event.team not in ("home", "away"):
            return (0.0, 0.0)

        favours_acting_team = event.event_type in _FAVOURS_ACTING_TEAM
        signed_weight = weight if favours_acting_team else -weight

        if event.team == "home":
            return (signed_weight, -signed_weight)
        return (-signed_weight, signed_weight)

    @staticmethod
    def _normalise_probabilities(
        home_prob: float, draw_prob: float, away_prob: float
    ) -> tuple[float, float, float]:
        """Clip probabilities to a sane range and renormalise so they sum to 1.0."""
        clipped = [
            max(_MIN_PROBABILITY, min(_MAX_PROBABILITY, p))
            for p in (home_prob, draw_prob, away_prob)
        ]
        total = sum(clipped)
        return tuple(p / total for p in clipped)  # type: ignore[return-value]

    @staticmethod
    def _validate_odds_bounds(odds: BettingOdds) -> None:
        """Ensure calculated odds fall within the platform's acceptable range.

        Raises:
            AnomalousOddsError: If any outcome's odds are outside
                [config.min_odds_value, config.max_odds_value].
        """
        for value in (odds.home_win, odds.draw, odds.away_win):
            if not (config.min_odds_value <= value <= config.max_odds_value):
                raise AnomalousOddsError(odds.match_id, value)

    @staticmethod
    def _check_suspicious_movement(previous: BettingOdds, new: BettingOdds) -> None:
        """Detect UKGC-reportable odds movement (>20% change in <60 seconds).

        Raises:
            SuspiciousActivityError: If the movement threshold is breached within
                the monitoring window. The caller is responsible for routing the
                match to manual compliance review.
        """
        elapsed_seconds = (datetime.utcnow() - previous.calculated_at).total_seconds()

        movements = [
            abs(new_value - old_value) / old_value
            for old_value, new_value in (
                (previous.home_win, new.home_win),
                (previous.draw, new.draw),
                (previous.away_win, new.away_win),
            )
            if old_value > 0
        ]
        max_movement = max(movements, default=0.0)

        if max_movement > config.suspicious_movement_threshold and elapsed_seconds < 60:
            logger.warning(
                "suspicious_odds_movement_detected",
                match_id=new.match_id,
                movement_pct=max_movement,
                elapsed_seconds=elapsed_seconds,
            )
            raise SuspiciousActivityError(new.match_id, max_movement, int(elapsed_seconds))

    @staticmethod
    def _build_audit_entry(
        event: MatchEvent,
        phase: str,
        before: BettingOdds | None,
        after: BettingOdds | None,
    ) -> dict:
        """Build a UKGC-compliant audit entry for an odds calculation.

        Includes timestamp (UTC ISO 8601), entity ID, before/after values,
        actor and reason for change, as required for regulatory audit.
        """
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "match_id": event.match_id,
            "market_type": MATCH_RESULT_MARKET,
            "phase": phase,
            "before_values": OddsCalculationService._odds_snapshot(before),
            "after_values": OddsCalculationService._odds_snapshot(after),
            "actor": "system:odds-calculation-service",
            "reason": f"{event.event_type.value} (team={event.team}, minute={event.minute})",
            "triggering_event_id": str(event.id),
        }

    @staticmethod
    def _odds_snapshot(odds: BettingOdds | None) -> dict | None:
        """Serialise a BettingOdds record for inclusion in an audit entry."""
        if odds is None:
            return None
        return {
            "odds_id": str(odds.id),
            "home_win": odds.home_win,
            "draw": odds.draw,
            "away_win": odds.away_win,
            "margin_applied": odds.margin_applied,
            "calculated_at": odds.calculated_at.isoformat() + "Z",
        }
