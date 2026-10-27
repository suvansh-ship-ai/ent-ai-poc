"""
Module: odds_calculation_service.py
Description: Production-ready OddsCalculationService for live football
    match odds recalculation. Receives match events (goals, red cards,
    injuries, substitutions) from the event stream, recalculates live
    odds for in-play betting markets, publishes updated odds to the
    trading platform, and logs all calculations for UKGC regulatory
    audit compliance.
Author: AI Virtual Engineer
Date: 2026-10-27
Jira: SCRUM-1
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SERVICE_NAME: str = "OddsCalculationService"
SERVICE_VERSION: str = "1.0.0"
MAX_ODDS_VALUE: Decimal = Decimal("1000.00")
MIN_ODDS_VALUE: Decimal = Decimal("1.01")
DEFAULT_HOME_WIN_ODDS: Decimal = Decimal("2.50")
DEFAULT_DRAW_ODDS: Decimal = Decimal("3.20")
DEFAULT_AWAY_WIN_ODDS: Decimal = Decimal("2.80")
GOAL_IMPACT_FACTOR: Decimal = Decimal("0.35")
RED_CARD_IMPACT_FACTOR: Decimal = Decimal("0.20")
INJURY_IMPACT_FACTOR: Decimal = Decimal("0.10")
SUBSTITUTION_IMPACT_FACTOR: Decimal = Decimal("0.05")

# ---------------------------------------------------------------------------
# Logging — structured audit logger for UKGC compliance
# ---------------------------------------------------------------------------
logger = logging.getLogger(SERVICE_NAME)
logger.setLevel(logging.INFO)

audit_logger = logging.getLogger(f"{SERVICE_NAME}.audit")
audit_logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------
class EntainBaseException(Exception):
    """Base exception for all Entain service errors."""

    def __init__(
        self,
        message: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class InvalidMatchEventError(EntainBaseException):
    """Raised when a match event payload is invalid or malformed."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="ERR_INVALID_MATCH_EVENT",
            details=details or {},
        )


class OddsCalculationError(EntainBaseException):
    """Raised when odds recalculation fails."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="ERR_ODDS_CALCULATION",
            details=details or {},
        )


class OddsPublishError(EntainBaseException):
    """Raised when publishing odds to the trading platform fails."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="ERR_ODDS_PUBLISH",
            details=details or {},
        )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class MatchEventType(Enum):
    """Supported match event types for odds recalculation."""

    GOAL = "goal"
    RED_CARD = "red_card"
    INJURY = "injury"
    SUBSTITUTION = "substitution"


class MarketType(Enum):
    """Supported betting market types."""

    MATCH_ODDS = "match_odds"
    OVER_UNDER = "over_under"
    BOTH_TEAMS_TO_SCORE = "both_teams_to_score"


class TeamSide(Enum):
    """Identifies which team the event applies to."""

    HOME = "home"
    AWAY = "away"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MatchEvent:
    """Represents a single in-play match event."""

    event_id: str
    match_id: str
    event_type: MatchEventType
    team_side: TeamSide
    minute: int
    player_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class MatchOdds:
    """Current odds state for a match."""

    match_id: str
    home_win: Decimal
    draw: Decimal
    away_win: Decimal
    market_type: MarketType = MarketType.MATCH_ODDS
    last_updated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    version: int = 1


@dataclass(frozen=True)
class OddsUpdate:
    """Published odds update payload for the trading platform."""

    update_id: str
    match_id: str
    previous_odds: Dict[str, str]
    new_odds: Dict[str, str]
    trigger_event_id: str
    market_type: str
    calculated_at: str
    version: int


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class OddsCalculationService:
    """Receives match events and recalculates live in-play odds.

    This service is the core odds engine for Entain's live football
    betting platform.  Every calculation is audit-logged in compliance
    with UKGC regulations.

    Attributes:
        _odds_store: In-memory store of current odds keyed by match_id.
        _event_history: Chronological list of processed events per match.
        _trading_platform: Adapter for publishing odds updates.
    """

    def __init__(
        self,
        trading_platform: Optional[Any] = None,
    ) -> None:
        """Initialise the OddsCalculationService.

        Args:
            trading_platform: Optional adapter/client for publishing
                odds to the downstream trading platform.  When *None*
                a default stub is used (useful for testing).
        """
        self._odds_store: Dict[str, MatchOdds] = {}
        self._event_history: Dict[str, List[MatchEvent]] = {}
        self._trading_platform = trading_platform
        logger.info(
            "OddsCalculationService initialised",
            extra={
                "service_version": SERVICE_VERSION,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_event(self, raw_event: Dict[str, Any]) -> OddsUpdate:
        """Process an incoming match event and return the odds update.

        This is the main entry-point called by the event-stream
        consumer.  It validates the event, recalculates odds, publishes
        the update, and performs UKGC audit logging.

        Args:
            raw_event: Raw event dictionary from the event stream.

        Returns:
            An ``OddsUpdate`` describing the change.

        Raises:
            InvalidMatchEventError: If the event payload is malformed.
            OddsCalculationError: If recalculation fails.
            OddsPublishError: If publishing to the trading platform
                fails.
        """
        start_time = time.monotonic()

        # 1. Validate & parse
        event = self._validate_and_parse(raw_event)

        # 2. Retrieve or initialise current odds
        current_odds = self._get_or_create_odds(event.match_id)

        # 3. Snapshot previous odds for audit trail
        previous_odds = self._snapshot_odds(current_odds)

        # 4. Recalculate
        new_odds = self._recalculate_odds(current_odds, event)

        # 5. Build update payload
        odds_update = self._build_odds_update(
            event=event,
            previous_odds=previous_odds,
            new_odds=new_odds,
        )

        # 6. Publish to trading platform
        self._publish_odds(odds_update)

        # 7. Persist new odds state
        self._odds_store[event.match_id] = new_odds
        self._event_history.setdefault(event.match_id, []).append(event)

        elapsed_ms = round((time.monotonic() - start_time) * 1000, 2)

        # 8. UKGC audit log
        self._audit_log_calculation(
            event=event,
            previous_odds=previous_odds,
            new_odds=new_odds,
            elapsed_ms=elapsed_ms,
        )

        logger.info(
            "Event processed successfully",
            extra={
                "match_id": event.match_id,
                "event_type": event.event_type.value,
                "elapsed_ms": elapsed_ms,
            },
        )

        return odds_update

    def get_current_odds(self, match_id: str) -> Optional[MatchOdds]:
        """Return the latest odds for a match, or *None*.

        Args:
            match_id: Unique identifier of the match.

        Returns:
            The current ``MatchOdds`` or *None* if no odds exist.
        """
        return self._odds_store.get(match_id)

    def get_event_history(
        self, match_id: str
    ) -> List[MatchEvent]:
        """Return the chronological event history for a match.

        Args:
            match_id: Unique identifier of the match.

        Returns:
            List of ``MatchEvent`` objects processed for the match.
        """
        return list(self._event_history.get(match_id, []))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_and_parse(
        self, raw_event: Dict[str, Any]
    ) -> MatchEvent:
        """Validate raw event dict and return a ``MatchEvent``.

        Args:
            raw_event: Dictionary from the event stream.

        Returns:
            Parsed ``MatchEvent``.

        Raises:
            InvalidMatchEventError: On missing/invalid fields.
        """
        required_fields = [
            "event_id",
            "match_id",
            "event_type",
            "team_side",
            "minute",
        ]
        missing = [
            f for f in required_fields if f not in raw_event
        ]
        if missing:
            raise InvalidMatchEventError(
                message=(
                    f"Missing required fields: {', '.join(missing)}"
                ),
                details={"raw_event": raw_event},
            )

        try:
            event_type = MatchEventType(raw_event["event_type"])
        except ValueError as exc:
            raise InvalidMatchEventError(
                message=(
                    f"Unsupported event_type: "
                    f"{raw_event['event_type']}"
                ),
                details={"raw_event": raw_event},
            ) from exc

        try:
            team_side = TeamSide(raw_event["team_side"])
        except ValueError as exc:
            raise InvalidMatchEventError(
                message=(
                    f"Invalid team_side: {raw_event['team_side']}"
                ),
                details={"raw_event": raw_event},
            ) from exc

        minute = raw_event["minute"]
        if not isinstance(minute, int) or minute < 0:
            raise InvalidMatchEventError(
                message=f"Invalid minute value: {minute}",
                details={"raw_event": raw_event},
            )

        return MatchEvent(
            event_id=str(raw_event["event_id"]),
            match_id=str(raw_event["match_id"]),
            event_type=event_type,
            team_side=team_side,
            minute=minute,
            player_name=raw_event.get("player_name"),
            metadata=raw_event.get("metadata", {}),
        )

    # ------------------------------------------------------------------
    # Odds helpers
    # ------------------------------------------------------------------

    def _get_or_create_odds(self, match_id: str) -> MatchOdds:
        """Retrieve existing odds or create defaults for a new match.

        Args:
            match_id: Unique match identifier.

        Returns:
            Current ``MatchOdds`` for the match.
        """
        if match_id in self._odds_store:
            return self._odds_store[match_id]

        default_odds = MatchOdds(
            match_id=match_id,
            home_win=DEFAULT_HOME_WIN_ODDS,
            draw=DEFAULT_DRAW_ODDS,
            away_win=DEFAULT_AWAY_WIN_ODDS,
        )
        self._odds_store[match_id] = default_odds
        logger.info(
            "Initialised default odds for match",
            extra={"match_id": match_id},
        )
        return default_odds

    @staticmethod
    def _snapshot_odds(odds: MatchOdds) -> Dict[str, str]:
        """Create a serialisable snapshot of current odds.

        Args:
            odds: The current ``MatchOdds``.

        Returns:
            Dictionary with string representations of each outcome.
        """
        return {
            "home_win": str(odds.home_win),
            "draw": str(odds.draw),
            "away_win": str(odds.away_win),
        }

    @staticmethod
    def _clamp_odds(value: Decimal) -> Decimal:
        """Clamp an odds value within the permitted range.

        Args:
            value: Raw calculated odds value.

        Returns:
            Clamped ``Decimal`` between ``MIN_ODDS_VALUE`` and
            ``MAX_ODDS_VALUE``.
        """
        return max(MIN_ODDS_VALUE, min(MAX_ODDS_VALUE, value))

    def _get_impact_factor(
        self, event_type: MatchEventType
    ) -> Decimal:
        """Return the odds impact factor for a given event type.

        Args:
            event_type: The type of match event.

        Returns:
            Decimal impact factor.
        """
        impact_map: Dict[MatchEventType, Decimal] = {
            MatchEventType.GOAL: GOAL_IMPACT_FACTOR,
            MatchEventType.RED_CARD: RED_CARD_IMPACT_FACTOR,
            MatchEventType.INJURY: INJURY_IMPACT_FACTOR,
            MatchEventType.SUBSTITUTION: SUBSTITUTION_IMPACT_FACTOR,
        }
        return impact_map.get(event_type, Decimal("0.00"))

    # ------------------------------------------------------------------
    # Core recalculation
    # ------------------------------------------------------------------

    def _recalculate_odds(
        self,
        current_odds: MatchOdds,
        event: MatchEvent,
    ) -> MatchOdds:
        """Recalculate odds based on the incoming match event.

        The algorithm applies an impact factor that adjusts the
        scoring team's win odds downward (more likely) and the
        opposing team's odds upward (less likely).  Draw odds are
        adjusted proportionally.

        Args:
            current_odds: The pre-event odds state.
            event: The match event triggering recalculation.

        Returns:
            Updated ``MatchOdds``.

        Raises:
            OddsCalculationError: If the calculation produces an
                invalid result.
        """
        try:
            impact = self._get_impact_factor(event.event_type)

            # Time-decay: events later in the match have more impact
            minute_factor = Decimal(str(event.minute)) / Decimal("90")
            time_weight = Decimal("1.0") + (
                minute_factor * Decimal("0.5")
            )
            adjusted_impact = impact * time_weight

            if event.team_side == TeamSide.HOME:
                new_home = self._clamp_odds(
                    current_odds.home_win
                    * (Decimal("1.0") - adjusted_impact)
                )
                new_away = self._clamp_odds(
                    current_odds.away_win
                    * (Decimal("1.0") + adjusted_impact)
                )
            else:
                new_home = self._clamp_odds(
                    current_odds.home_win
                    * (Decimal("1.0") + adjusted_impact)
                )
                new_away = self._clamp_odds(
                    current_odds.away_win
                    * (Decimal("1.0") - adjusted_impact)
                )

            # Draw odds move toward the middle
            draw_shift = adjusted_impact * Decimal("0.5")
            new_draw = self._clamp_odds(
                current_odds.draw
                * (Decimal("1.0") + draw_shift)
            )

            return MatchOdds(
                match_id=current_odds.match_id,
                home_win=new_home.quantize(Decimal("0.01")),
                draw=new_draw.quantize(Decimal("0.01")),
                away_win=new_away.quantize(Decimal("0.01")),
                market_type=current_odds.market_type,
                version=current_odds.version + 1,
            )

        except (InvalidOperation, ArithmeticError) as exc:
            raise OddsCalculationError(
                message=f"Odds recalculation failed: {exc}",
                details={
                    "match_id": event.match_id,
                    "event_id": event.event_id,
                },
            ) from exc

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _build_odds_update(
        self,
        event: MatchEvent,
        previous_odds: Dict[str, str],
        new_odds: MatchOdds,
    ) -> OddsUpdate:
        """Build an ``OddsUpdate`` payload for the trading platform.

        Args:
            event: The triggering match event.
            previous_odds: Snapshot of odds before recalculation.
            new_odds: The newly calculated ``MatchOdds``.

        Returns:
            An ``OddsUpdate`` ready for publishing.
        """
        return OddsUpdate(
            update_id=str(uuid.uuid4()),
            match_id=event.match_id,
            previous_odds=previous_odds,
            new_odds=self._snapshot_odds(new_odds),
            trigger_event_id=event.event_id,
            market_type=new_odds.market_type.value,
            calculated_at=datetime.now(timezone.utc).isoformat(),
            version=new_odds.version,
        )

    def _publish_odds(self, odds_update: OddsUpdate) -> None:
        """Publish the odds update to the trading platform.

        Args:
            odds_update: The update payload to publish.

        Raises:
            OddsPublishError: If the trading platform rejects or
                fails to accept the update.
        """
        try:
            if self._trading_platform is not None:
                self._trading_platform.publish(
                    odds_update.__dict__
                )
            else:
                logger.info(
                    "Trading platform stub — odds update logged",
                    extra={
                        "update_id": odds_update.update_id,
                        "match_id": odds_update.match_id,
                    },
                )

            # UKGC audit log for publish event
            audit_logger.info(
                "Odds published to trading platform",
                extra={
                    "audit": True,
                    "event_type": "odds_published",
                    "update_id": odds_update.update_id,
                    "match_id": odds_update.match_id,
                    "market_type": odds_update.market_type,
                    "new_odds": odds_update.new_odds,
                    "version": odds_update.version,
                    "published_at": odds_update.calculated_at,
                },
            )

        except Exception as exc:
            raise OddsPublishError(
                message=(
                    f"Failed to publish odds update "
                    f"{odds_update.update_id}: {exc}"
                ),
                details={
                    "update_id": odds_update.update_id,
                    "match_id": odds_update.match_id,
                },
            ) from exc

    # ------------------------------------------------------------------
    # UKGC Audit Logging
    # ------------------------------------------------------------------

    def _audit_log_calculation(
        self,
        event: MatchEvent,
        previous_odds: Dict[str, str],
        new_odds: MatchOdds,
        elapsed_ms: float,
    ) -> None:
        """Write a UKGC-compliant audit log entry for the calculation.

        Args:
            event: The triggering match event.
            previous_odds: Odds snapshot before recalculation.
            new_odds: The newly calculated odds.
            elapsed_ms: Processing time in milliseconds.
        """
        audit_logger.info(
            "Odds recalculation audit entry",
            extra={
                "audit": True,
                "event_type": "odds_recalculation",
                "service": SERVICE_NAME,
                "service_version": SERVICE_VERSION,
                "match_id": event.match_id,
                "event_id": event.event_id,
                "match_event_type": event.event_type.value,
                "team_side": event.team_side.value,
                "minute": event.minute,
                "player_name": event.player_name,
                "previous_odds": previous_odds,
                "new_odds": {
                    "home_win": str(new_odds.home_win),
                    "draw": str(new_odds.draw),
                    "away_win": str(new_odds.away_win),
                },
                "odds_version": new_odds.version,
                "elapsed_ms": elapsed_ms,
                "calculated_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            },
        )
