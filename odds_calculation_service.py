"""Entain Sports Betting Platform — Odds Calculation Service.

Handles live odds recalculation for in-play football markets based on
match events (goals, red cards, injuries, substitutions, etc.), publishes
updated odds to the trading platform, and produces UKGC-compliant audit
logs for every odds calculation/change.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Protocol
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


# ═══════════════════════════════════════════════════════════════
# COLLABORATOR INTERFACES (Repository / Client Protocols)
# ═══════════════════════════════════════════════════════════════

class OddsRepository(Protocol):
    """Persistence layer for betting odds records."""

    async def get_latest_odds(self, match_id: str, market_type: str) -> BettingOdds | None:
        """Return the most recently calculated odds for a match/market, if any."""
        ...

    async def save(self, odds: BettingOdds) -> None:
        """Persist a newly calculated odds record. MUST be append-only (never update/delete)."""
        ...

    async def get_market_status(self, match_id: str, market_type: str) -> MarketStatus:
        """Return the current trading status of a market."""
        ...


class TradingPlatformClient(Protocol):
    """Client for publishing odds updates to the trading platform."""

    async def publish_odds(self, odds: BettingOdds) -> None:
        """Publish updated odds to the trading platform for customer-facing display."""
        ...


class AuditLogRepository(Protocol):
    """Append-only audit trail for regulatory (UKGC) reporting."""

    async def record_odds_change(self, audit_entry: dict[str, Any]) -> None:
        """Persist an immutable audit record of an odds calculation/change."""
        ...


# ═══════════════════════════════════════════════════════════════
# SERVICE
# ═══════════════════════════════════════════════════════════════

class OddsCalculationService:
    """Recalculates live in-play odds in response to match events.

    Responsibilities:
        * Consume match events (goals, red cards, injuries, substitutions, etc.).
        * Recalculate odds for the affected market using a base-price + margin model.
        * Reject anomalous odds outside the configured bounds.
        * Detect and flag suspicious odds movements (UKGC requirement: >20% in <60s).
        * Publish updated odds to the trading platform.
        * Write an immutable audit log entry for every calculation (UKGC requirement).

    Services are stateless and injectable — all I/O is delegated to the
    injected repository/client collaborators (repository pattern).
    """

    #: Relative "market shock" applied to the side that benefits from each event type.
    _EVENT_IMPACT: dict[MatchEventType, float] = {
        MatchEventType.GOAL_SCORED: 0.35,
        MatchEventType.RED_CARD: 0.20,
        MatchEventType.PENALTY_AWARDED: 0.15,
        MatchEventType.VAR_DECISION: 0.10,
        MatchEventType.INJURY: 0.05,
        MatchEventType.SUBSTITUTION: 0.02,
        MatchEventType.YELLOW_CARD: 0.01,
        MatchEventType.HALF_TIME: 0.0,
        MatchEventType.FULL_TIME: 0.0,
    }

    #: Default baseline odds used when no previous odds exist for a market yet.
    _DEFAULT_BASELINE = {"home_win": 2.50, "draw": 3.20, "away_win": 2.80}

    _MAX_RETRIES = 3
    _RETRY_BACKOFF_BASE_SECONDS = 0.5

    def __init__(
        self,
        odds_repo: OddsRepository,
        trading_platform: TradingPlatformClient,
        audit_log: AuditLogRepository,
        margin_pct: float | None = None,
    ) -> None:
        """Initialise the service with its collaborators.

        Args:
            odds_repo: Repository for reading/writing odds records.
            trading_platform: Client used to publish odds to the trading platform.
            audit_log: Append-only audit trail repository (UKGC requirement).
            margin_pct: Override for the bookmaker margin. Defaults to `config.default_margin_pct`.
        """
        self._odds = odds_repo
        self._trading = trading_platform
        self._audit = audit_log
        self._margin_pct = margin_pct if margin_pct is not None else config.default_margin_pct

    async def calculate_live_odds(self, event: MatchEvent, market_type: str = "match_result") -> BettingOdds:
        """Calculate, validate, persist, and publish updated odds for a match event.

        This is the main entry point invoked when a match event (goal, red card,
        injury, substitution, etc.) arrives from the event stream.

        Args:
            event: The match event received from the event stream.
            market_type: The market to recalculate (defaults to ``"match_result"``).

        Returns:
            The newly calculated and published `BettingOdds` record.

        Raises:
            OddsCalculationError: If odds calculation, persistence, or publishing fails.
            MarketSuspendedError: If the target market is not open for trading.
            AnomalousOddsError: If the calculated odds fall outside acceptable bounds.
            SuspiciousActivityError: If the odds movement exceeds the UKGC-reportable threshold.
        """
        log = logger.bind(
            match_id=event.match_id,
            event_id=str(event.id),
            event_type=event.event_type.value,
            market_type=market_type,
        )

        market_status = await self._call_with_retry(
            self._odds.get_market_status,
            event.match_id,
            market_type,
            op_name="get_market_status",
            match_id=event.match_id,
        )

        if market_status != MarketStatus.OPEN:
            log.warning("odds_calculation_blocked_market_not_open", market_status=market_status.value)
            raise MarketSuspendedError(event.match_id, market_type)

        previous_odds = await self._call_with_retry(
            self._odds.get_latest_odds,
            event.match_id,
            market_type,
            op_name="get_latest_odds",
            match_id=event.match_id,
        )

        # UKGC requirement: log BEFORE performing the calculation/financial operation.
        log.info(
            "odds_calculation_started",
            previous_odds=self._odds_snapshot(previous_odds),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        new_odds = self._apply_event_impact(event, previous_odds, market_type)
        self._validate_odds_bounds(new_odds)

        movement_pct = self._movement_pct(previous_odds, new_odds)
        if previous_odds is not None and movement_pct >= config.suspicious_movement_threshold:
            window_seconds = int((new_odds.calculated_at - previous_odds.calculated_at).total_seconds())
            if window_seconds < 60:
                log.warning(
                    "suspicious_odds_movement_detected",
                    movement_pct=movement_pct,
                    window_seconds=window_seconds,
                )
                await self._record_audit(
                    event=event,
                    market_type=market_type,
                    previous_odds=previous_odds,
                    new_odds=new_odds,
                    reason="suspicious_movement_alert",
                )
                raise SuspiciousActivityError(event.match_id, movement_pct, window_seconds)

        await self._call_with_retry(
            self._odds.save,
            new_odds,
            op_name="save_odds",
            match_id=event.match_id,
        )

        await self._call_with_retry(
            self._trading.publish_odds,
            new_odds,
            op_name="publish_odds",
            match_id=event.match_id,
        )

        # UKGC requirement: log AFTER the operation with before/after values.
        await self._record_audit(
            event=event,
            market_type=market_type,
            previous_odds=previous_odds,
            new_odds=new_odds,
            reason=f"match_event:{event.event_type.value}",
        )

        log.info("odds_calculation_completed", new_odds=self._odds_snapshot(new_odds))

        return new_odds

    def _apply_event_impact(
        self,
        event: MatchEvent,
        previous_odds: BettingOdds | None,
        market_type: str,
    ) -> BettingOdds:
        """Compute new odds from the previous odds and the event's market impact.

        Args:
            event: The triggering match event.
            previous_odds: The previous odds record, or None if none exists yet.
            market_type: The market being recalculated.

        Returns:
            A new `BettingOdds` instance (not yet persisted).

        Raises:
            OddsCalculationError: If no baseline odds exist and the event alone
                cannot establish a starting price.
        """
        base_home = previous_odds.home_win if previous_odds else self._DEFAULT_BASELINE["home_win"]
        base_draw = previous_odds.draw if previous_odds else self._DEFAULT_BASELINE["draw"]
        base_away = previous_odds.away_win if previous_odds else self._DEFAULT_BASELINE["away_win"]

        impact = self._EVENT_IMPACT.get(event.event_type, 0.0)

        if impact == 0.0 and previous_odds is None:
            raise OddsCalculationError(event.match_id, "no baseline odds and event has no market impact")

        # The team benefiting from the event shortens; the opposing side lengthens.
        if event.team == "home":
            home = max(base_home * (1 - impact), config.min_odds_value)
            away = base_away * (1 + impact)
        elif event.team == "away":
            away = max(base_away * (1 - impact), config.min_odds_value)
            home = base_home * (1 + impact)
        else:
            home, away = base_home, base_away

        draw = base_draw * (1 + impact * 0.5)

        home, draw, away = self._apply_margin(home, draw, away)

        return BettingOdds(
            match_id=event.match_id,
            market_type=market_type,
            home_win=round(home, 2),
            draw=round(draw, 2),
            away_win=round(away, 2),
            margin_applied=self._margin_pct,
            calculated_at=datetime.now(timezone.utc),
            previous_odds_id=previous_odds.id if previous_odds else None,
        )

    def _apply_margin(self, home: float, draw: float, away: float) -> tuple[float, float, float]:
        """Apply the bookmaker margin evenly across all outcomes.

        Args:
            home: Raw (fair) home win odds.
            draw: Raw (fair) draw odds.
            away: Raw (fair) away win odds.

        Returns:
            Tuple of (home, draw, away) odds with margin applied.
        """
        margin_factor = 1 - self._margin_pct
        return home * margin_factor, draw * margin_factor, away * margin_factor

    def _validate_odds_bounds(self, odds: BettingOdds) -> None:
        """Validate that all outcomes fall within acceptable odds bounds.

        Args:
            odds: The calculated odds to validate.

        Raises:
            AnomalousOddsError: If any outcome is outside [min_odds_value, max_odds_value].
        """
        for value in (odds.home_win, odds.draw, odds.away_win):
            if value < config.min_odds_value or value > config.max_odds_value:
                raise AnomalousOddsError(odds.match_id, value)

    @staticmethod
    def _movement_pct(previous: BettingOdds | None, new: BettingOdds) -> float:
        """Calculate the largest percentage movement across all outcomes.

        Args:
            previous: The previous odds record, or None.
            new: The newly calculated odds record.

        Returns:
            The largest absolute percentage change across home/draw/away, or 0.0
            if there is no previous record to compare against.
        """
        if previous is None:
            return 0.0

        movements: list[float] = []
        for prev_value, new_value in (
            (previous.home_win, new.home_win),
            (previous.draw, new.draw),
            (previous.away_win, new.away_win),
        ):
            if prev_value:
                movements.append(abs(new_value - prev_value) / prev_value)

        return max(movements, default=0.0)

    async def _record_audit(
        self,
        event: MatchEvent,
        market_type: str,
        previous_odds: BettingOdds | None,
        new_odds: BettingOdds,
        reason: str,
    ) -> None:
        """Persist an immutable audit record of an odds change (UKGC requirement).

        The audit entry captures timestamp (UTC ISO 8601), entity ID, before/after
        values, actor, and reason for change as mandated by UKGC regulatory standards.

        Args:
            event: The triggering match event.
            market_type: The market that was recalculated.
            previous_odds: The odds before this change, if any.
            new_odds: The odds after this change.
            reason: Human-readable reason for the change.

        Raises:
            OddsCalculationError: If the audit record cannot be persisted. Audit
                failures must never be swallowed silently (regulatory requirement).
        """
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "match_id": event.match_id,
            "market_type": market_type,
            "event_id": str(event.id),
            "before": self._odds_snapshot(previous_odds),
            "after": self._odds_snapshot(new_odds),
            "actor": "system:odds_calculation_service",
            "reason": reason,
        }

        try:
            await self._audit.record_odds_change(audit_entry)
        except Exception as exc:  # noqa: BLE001 - wrap all external errors
            logger.error("audit_log_write_failed", match_id=event.match_id, error=str(exc))
            raise OddsCalculationError(event.match_id, f"failed to write audit log: {exc}") from exc

    @staticmethod
    def _odds_snapshot(odds: BettingOdds | None) -> dict[str, Any] | None:
        """Serialise a `BettingOdds` record into a plain dict for logging/audit purposes.

        Args:
            odds: The odds record to snapshot, or None.

        Returns:
            A dict snapshot, or None if `odds` is None.
        """
        if odds is None:
            return None

        return {
            "id": str(odds.id),
            "home_win": odds.home_win,
            "draw": odds.draw,
            "away_win": odds.away_win,
            "margin_applied": odds.margin_applied,
            "calculated_at": odds.calculated_at.isoformat(),
        }

    async def _call_with_retry(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        op_name: str,
        match_id: str,
        **kwargs: Any,
    ) -> Any:
        """Invoke an external async collaborator with retry + exponential backoff.

        Wraps every call to an injected repository/client so that transient
        failures (network blips, connection resets) are retried up to
        `_MAX_RETRIES` times with exponential backoff before surfacing as an
        `OddsCalculationError`.

        Args:
            func: The async callable to invoke.
            *args: Positional arguments passed to `func`.
            op_name: Name of the operation, used for logging context.
            match_id: The match ID, used for logging context and error reporting.
            **kwargs: Keyword arguments passed to `func`.

        Returns:
            The result of `func(*args, **kwargs)`.

        Raises:
            OddsCalculationError: If all retry attempts are exhausted.
        """
        last_exc: Exception | None = None

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - wrap all external errors
                last_exc = exc
                logger.warning(
                    "external_call_failed_retrying",
                    operation=op_name,
                    match_id=match_id,
                    attempt=attempt,
                    max_attempts=self._MAX_RETRIES,
                    error=str(exc),
                )
                if attempt < self._MAX_RETRIES:
                    await asyncio.sleep(self._RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))

        raise OddsCalculationError(match_id, f"{op_name} failed after {self._MAX_RETRIES} attempts: {last_exc}")
