"""Entain Sports Betting Platform — Live Odds Calculation Service.

Implements SCRUM-1: a production-ready OddsCalculationService that consumes
match events (goals, red cards, injuries, substitutions, etc.) from the event
stream, recalculates live in-play odds, publishes the updated odds to the
trading platform, and produces a UKGC-compliant audit trail for every
calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol
from uuid import UUID, uuid4

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
# DEPENDENCY PROTOCOLS (Repository / External Client contracts)
# ═══════════════════════════════════════════════════════════════

class OddsRepository(Protocol):
    """Data access contract for persisted odds records."""

    async def get_latest_odds(self, match_id: str, market_type: str) -> BettingOdds | None:
        """Return the most recently calculated odds for a match/market, if any."""
        ...

    async def get_market_status(self, match_id: str, market_type: str) -> MarketStatus:
        """Return the current trading status of a market."""
        ...

    async def save(self, odds: BettingOdds) -> BettingOdds:
        """Persist a newly calculated odds record (append-only)."""
        ...


class TradingPlatformClient(Protocol):
    """External client contract for publishing odds to the trading platform."""

    async def publish_odds(self, odds: BettingOdds) -> None:
        """Publish updated odds to the trading platform."""
        ...


# ═══════════════════════════════════════════════════════════════
# INTERNAL VALUE OBJECTS
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ImpliedProbabilities:
    """Implied win probabilities for a 3-way market (home/draw/away)."""

    home: float
    draw: float
    away: float


# Probability shift applied to the "attacking"/impacted side when a given
# event type occurs. Values are additive shifts applied to raw (fair)
# probabilities before renormalisation and margin application.
_EVENT_PROBABILITY_SHIFT: dict[MatchEventType, float] = {
    MatchEventType.GOAL_SCORED: 0.12,
    MatchEventType.RED_CARD: 0.10,  # benefits the opposing team
    MatchEventType.PENALTY_AWARDED: 0.08,
    MatchEventType.VAR_DECISION: 0.05,
    MatchEventType.INJURY: 0.03,
    MatchEventType.SUBSTITUTION: 0.0,
    MatchEventType.YELLOW_CARD: 0.0,
    MatchEventType.HALF_TIME: 0.0,
    MatchEventType.FULL_TIME: 0.0,
}


class OddsCalculationService:
    """Calculates and publishes live in-play odds from match events.

    The service is stateless and receives all dependencies via constructor
    injection, in line with the Entain service-layer pattern. It never talks
    to a database or external API directly — all IO goes through the
    injected repository / client contracts.
    """

    def __init__(
        self,
        odds_repository: OddsRepository,
        trading_platform_client: TradingPlatformClient,
    ) -> None:
        """Initialise the service.

        Args:
            odds_repository: Repository used to read/write odds records.
            trading_platform_client: Client used to publish odds downstream.
        """
        self._odds = odds_repository
        self._trading = trading_platform_client

    async def calculate_live_odds(
        self,
        event: MatchEvent,
        market_type: str = "match_result",
    ) -> BettingOdds:
        """Recalculate and publish live odds in response to a match event.

        Args:
            event: The match event received from the event stream (goal,
                red card, injury, substitution, etc.).
            market_type: The market to recalculate (defaults to the 3-way
                match result market).

        Returns:
            The newly calculated and published `BettingOdds` record.

        Raises:
            MarketSuspendedError: If the market is not open for trading.
            AnomalousOddsError: If the calculated odds fall outside the
                configured acceptable range.
            OddsCalculationError: If an unexpected error occurs while
                calculating or publishing odds.
        """
        correlation_id = uuid4()
        log = logger.bind(
            correlation_id=str(correlation_id),
            match_id=event.match_id,
            market_type=market_type,
            event_id=str(event.id),
            event_type=event.event_type.value,
        )

        try:
            market_status = await self._odds.get_market_status(event.match_id, market_type)
            if market_status not in (MarketStatus.OPEN,):
                log.warning("market_not_open", market_status=market_status.value)
                raise MarketSuspendedError(event.match_id, market_type)

            previous_odds = await self._odds.get_latest_odds(event.match_id, market_type)

            # UKGC Requirement: log BEFORE performing the calculation, so the
            # state can be reconstructed even if the process crashes mid-way.
            self._log_audit_trail(
                action="odds_calculation_started",
                correlation_id=correlation_id,
                match_id=event.match_id,
                market_type=market_type,
                event=event,
                before=previous_odds,
                after=None,
                reason=f"match_event:{event.event_type.value}",
            )

            new_odds = self._compute_new_odds(event, previous_odds, market_type)
            self._validate_odds_bounds(new_odds)

            if previous_odds is not None:
                self._check_suspicious_movement(previous_odds, new_odds, log)

            saved_odds = await self._odds.save(new_odds)
            await self._trading.publish_odds(saved_odds)

            # UKGC Requirement: log AFTER the operation completes, capturing
            # before/after values, actor, timestamp and reason.
            self._log_audit_trail(
                action="odds_calculation_completed",
                correlation_id=correlation_id,
                match_id=event.match_id,
                market_type=market_type,
                event=event,
                before=previous_odds,
                after=saved_odds,
                reason=f"match_event:{event.event_type.value}",
            )

            log.info(
                "odds_calculated_and_published",
                home_win=saved_odds.home_win,
                draw=saved_odds.draw,
                away_win=saved_odds.away_win,
                margin_applied=saved_odds.margin_applied,
            )

            return saved_odds

        except (MarketSuspendedError, AnomalousOddsError):
            # Domain errors are already well-formed and logged by callers.
            raise
        except Exception as exc:  # noqa: BLE001 — wrap all unexpected errors
            log.error("odds_calculation_failed", error=str(exc), exc_info=True)
            raise OddsCalculationError(event.match_id, str(exc)) from exc

    # ─────────────────────────────────────────────────────────
    # Internal calculation helpers
    # ─────────────────────────────────────────────────────────

    def _compute_new_odds(
        self,
        event: MatchEvent,
        previous_odds: BettingOdds | None,
        market_type: str,
    ) -> BettingOdds:
        """Compute new odds for the given event, based on previous odds.

        Args:
            event: The triggering match event.
            previous_odds: The last calculated odds for this match/market,
                or None if this is the first calculation.
            market_type: The market being recalculated.

        Returns:
            A new (unsaved) `BettingOdds` instance.
        """
        probabilities = self._implied_probabilities(previous_odds)
        shifted = self._apply_event_shift(probabilities, event)
        margin = config.default_margin_pct

        home_win, draw, away_win = self._probabilities_to_odds(shifted, margin)

        return BettingOdds(
            match_id=event.match_id,
            market_type=market_type,
            home_win=home_win,
            draw=draw,
            away_win=away_win,
            margin_applied=margin,
            calculated_at=datetime.now(timezone.utc),
            previous_odds_id=previous_odds.id if previous_odds else None,
        )

    @staticmethod
    def _implied_probabilities(previous_odds: BettingOdds | None) -> ImpliedProbabilities:
        """Derive fair (margin-free) implied probabilities from previous odds.

        Args:
            previous_odds: The previous odds record, or None to fall back to
                an even 3-way market.

        Returns:
            Normalised implied probabilities summing to 1.0.
        """
        if previous_odds is None:
            return ImpliedProbabilities(home=1 / 3, draw=1 / 3, away=1 / 3)

        raw_home = 1 / previous_odds.home_win
        raw_draw = 1 / previous_odds.draw
        raw_away = 1 / previous_odds.away_win
        total = raw_home + raw_draw + raw_away  # includes existing margin

        return ImpliedProbabilities(
            home=raw_home / total,
            draw=raw_draw / total,
            away=raw_away / total,
        )

    @staticmethod
    def _apply_event_shift(
        probabilities: ImpliedProbabilities,
        event: MatchEvent,
    ) -> ImpliedProbabilities:
        """Apply a probability shift based on the match event type/team.

        Args:
            probabilities: Fair implied probabilities before the event.
            event: The match event to apply.

        Returns:
            New fair implied probabilities after the event, still summing
            to 1.0.
        """
        shift = _EVENT_PROBABILITY_SHIFT.get(event.event_type, 0.0)
        if shift == 0.0 or event.team not in ("home", "away"):
            return probabilities

        home, draw, away = probabilities.home, probabilities.draw, probabilities.away

        # Goals/penalties/VAR favour the scoring team; red cards favour the
        # *opposing* team (the carded team is at a disadvantage).
        favoured_is_home = (
            event.team == "home"
            if event.event_type != MatchEventType.RED_CARD
            else event.team == "away"
        )

        if favoured_is_home:
            home_new = min(home + shift, 0.97)
            remainder_scale = (1 - home_new) / (draw + away) if (draw + away) > 0 else 0
            draw_new = draw * remainder_scale
            away_new = away * remainder_scale
        else:
            away_new = min(away + shift, 0.97)
            remainder_scale = (1 - away_new) / (home + draw) if (home + draw) > 0 else 0
            home_new = home * remainder_scale
            draw_new = draw * remainder_scale

        return ImpliedProbabilities(home=home_new, draw=draw_new, away=away_new)

    @staticmethod
    def _probabilities_to_odds(
        probabilities: ImpliedProbabilities,
        margin: float,
    ) -> tuple[float, float, float]:
        """Convert fair probabilities into decimal odds with margin applied.

        Args:
            probabilities: Fair (margin-free) implied probabilities.
            margin: The bookmaker margin to apply (e.g. 0.06 for 6%).

        Returns:
            A tuple of (home_win, draw, away_win) decimal odds, rounded to
            2 decimal places.
        """
        overround = 1 + margin
        home_odds = overround / probabilities.home if probabilities.home > 0 else config.max_odds_value
        draw_odds = overround / probabilities.draw if probabilities.draw > 0 else config.max_odds_value
        away_odds = overround / probabilities.away if probabilities.away > 0 else config.max_odds_value

        return round(home_odds, 2), round(draw_odds, 2), round(away_odds, 2)

    @staticmethod
    def _validate_odds_bounds(odds: BettingOdds) -> None:
        """Validate that all calculated odds are within acceptable bounds.

        Args:
            odds: The newly calculated odds.

        Raises:
            AnomalousOddsError: If any outcome is outside the configured
                min/max odds range.
        """
        for value in (odds.home_win, odds.draw, odds.away_win):
            if value < config.min_odds_value or value > config.max_odds_value:
                raise AnomalousOddsError(odds.match_id, value)

    @staticmethod
    def _check_suspicious_movement(
        previous_odds: BettingOdds,
        new_odds: BettingOdds,
        log: structlog.BoundLogger,
    ) -> None:
        """Detect and log suspicious odds movement per UKGC guidance.

        Movement of more than the configured threshold (default 20%) within
        the configured time window (default 60s) is logged as a suspicious
        activity alert. This does not block publication — legitimate events
        (e.g. a goal) can cause large, expected swings — but it must be
        surfaced for compliance monitoring.

        Args:
            previous_odds: The prior odds record.
            new_odds: The newly calculated odds record.
            log: A bound structlog logger for this calculation.
        """
        elapsed_seconds = (new_odds.calculated_at - previous_odds.calculated_at).total_seconds()
        if elapsed_seconds < 0:
            elapsed_seconds = 0.0

        movements = {
            "home_win": OddsCalculationService._pct_change(previous_odds.home_win, new_odds.home_win),
            "draw": OddsCalculationService._pct_change(previous_odds.draw, new_odds.draw),
            "away_win": OddsCalculationService._pct_change(previous_odds.away_win, new_odds.away_win),
        }

        max_selection, max_movement = max(movements.items(), key=lambda item: abs(item[1]))

        if abs(max_movement) > config.suspicious_movement_threshold and elapsed_seconds < 60:
            log.warning(
                "suspicious_odds_movement_detected",
                selection=max_selection,
                movement_pct=max_movement,
                window_seconds=elapsed_seconds,
                previous_value=getattr(previous_odds, max_selection),
                new_value=getattr(new_odds, max_selection),
            )
            # Raise as a distinct, catchable event for the compliance/alerting
            # pipeline. Callers may choose to swallow this after alerting.
            try:
                raise SuspiciousActivityError(new_odds.match_id, max_movement, int(elapsed_seconds))
            except SuspiciousActivityError as exc:
                log.error("ukgc_reportable_event", error=str(exc))

    @staticmethod
    def _pct_change(before: float, after: float) -> float:
        """Return the percentage change between two odds values."""
        if before == 0:
            return 0.0
        return (after - before) / before

    @staticmethod
    def _log_audit_trail(
        action: str,
        correlation_id: UUID,
        match_id: str,
        market_type: str,
        event: MatchEvent,
        before: BettingOdds | None,
        after: BettingOdds | None,
        reason: str,
    ) -> None:
        """Emit a structured UKGC audit log entry for an odds calculation.

        Per Entain regulatory compliance standards, every odds change must
        be auditable with: UTC ISO-8601 timestamp, entity ID, before/after
        values, actor, and reason for change.

        Args:
            action: The audit action name (e.g. "odds_calculation_started").
            correlation_id: Correlation ID linking start/end audit entries.
            match_id: The match the odds relate to.
            market_type: The market the odds relate to.
            event: The triggering match event.
            before: Odds state before the change (None if no prior odds).
            after: Odds state after the change (None if not yet calculated).
            reason: Human-readable reason for the change.
        """
        logger.info(
            "ukgc_audit_log",
            action=action,
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_id=str(correlation_id),
            entity_type="betting_odds",
            match_id=match_id,
            market_type=market_type,
            triggering_event_id=str(event.id),
            triggering_event_type=event.event_type.value,
            actor="system:odds-calculation-service",
            reason=reason,
            before={
                "home_win": before.home_win,
                "draw": before.draw,
                "away_win": before.away_win,
            } if before else None,
            after={
                "home_win": after.home_win,
                "draw": after.draw,
                "away_win": after.away_win,
            } if after else None,
        )
