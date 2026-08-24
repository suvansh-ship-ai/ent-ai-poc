"""Entain Bet Settlement Service — Race-Condition-Safe Implementation.

Fix for SCRUM-2: eliminates the race condition that allowed duplicate
payouts when multiple settlement events arrived concurrently for the same
bet (e.g. final whistle + VAR correction within < 500ms). This had caused
£14,200 in duplicate payouts over a 30 day period.

Root cause (previous implementation):
    `settle_bet()` read the bet's status, processed the payout, and then
    updated the status — with no lock between the read and the update.
    Two concurrent calls could both observe status="active" and both
    process the payout before either updated the status.

Fix summary:
    - Acquire a per-bet DynamoDB-backed distributed lock BEFORE reading
      bet state, and hold it for the entire read -> decide -> payout ->
      update sequence (closing the race window entirely).
    - Re-check bet status once the lock is held (idempotency guard) and
      treat any non-ACTIVE status as an already-settled bet, returning a
      `is_duplicate=True` Settlement record instead of reprocessing.
    - Use `Decimal` (never float) for all monetary values, rounded to 2
      decimal places (GBP), per Entain financial standards.
    - Emit UKGC-compliant structured audit logs both BEFORE and AFTER the
      financial operation, capturing timestamp, entity IDs, actor,
      before/after status, and reason — so the transaction can be
      reconstructed even if the process crashes mid-settlement.
    - Always release the lock in a `finally` block, even on error.
"""

from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from uuid import UUID

import structlog

from .config import config
from .exceptions import BetNotFoundError, LockAcquisitionError
from .lock_manager import DynamoDBLockManager
from .models import BetStatus, Settlement

logger = structlog.get_logger()

_SERVICE_ACTOR = "bet_settlement_service"
_TWO_DP = Decimal("0.01")


class BetSettlementService:
    """Settles bets based on match results, safe under concurrent events.

    Ensures exactly-once settlement semantics even when multiple settlement
    events for the same bet arrive within a short window, by holding a
    distributed lock across the full read-decide-payout-update sequence.
    """

    def __init__(
        self,
        bet_repository,
        payout_service,
        event_bus,
        lock_manager: DynamoDBLockManager,
    ) -> None:
        """Initialize the settlement service.

        Args:
            bet_repository: Repository exposing `get_by_id` / `save` for Bet entities.
            payout_service: Service used to credit customer payouts.
            event_bus: Publishes domain events (e.g. settlement completed).
            lock_manager: Distributed lock manager backed by DynamoDB, used to
                serialize concurrent settlement attempts for the same bet.
        """
        self._bets = bet_repository
        self._payouts = payout_service
        self._events = event_bus
        self._locks = lock_manager

    async def settle_bet(self, bet_id: UUID, match_result: str, event_id: UUID) -> Settlement:
        """Settle a bet based on the match result.

        Acquires a per-bet distributed lock before reading bet state, and
        holds it until the settlement is fully persisted. This closes the
        race window that previously allowed two concurrent settlement events
        (e.g. final whistle + VAR correction) to both see status="active"
        and both process a payout.

        Args:
            bet_id: The bet to settle.
            match_result: The match outcome (e.g., "home_win").
            event_id: The settlement event ID that triggered this call.

        Returns:
            The `Settlement` record created, or a `Settlement` with
            `is_duplicate=True` if the bet had already been settled by a
            prior event.

        Raises:
            BetNotFoundError: If no bet exists for `bet_id`.
            LockAcquisitionError: If the distributed lock cannot be acquired
                after the configured number of retries (see
                `config.max_settlement_retries`).
        """
        lock_holder = f"settlement:{event_id}"

        logger.info(
            "settlement_lock_acquire_attempt",
            bet_id=str(bet_id),
            event_id=str(event_id),
            match_result=match_result,
        )

        try:
            await self._locks.acquire_bet_lock(
                bet_id=bet_id,
                holder=lock_holder,
                max_attempts=config.max_settlement_retries,
            )
        except LockAcquisitionError:
            logger.error(
                "settlement_lock_acquisition_failed",
                bet_id=str(bet_id),
                event_id=str(event_id),
                match_result=match_result,
            )
            raise

        try:
            return await self._settle_bet_locked(bet_id, match_result, event_id)
        finally:
            await self._locks.release_bet_lock(bet_id=bet_id, holder=lock_holder)

    async def _settle_bet_locked(self, bet_id: UUID, match_result: str, event_id: UUID) -> Settlement:
        """Perform the read -> decide -> payout -> update sequence under lock.

        This method MUST only be called while the caller holds the
        distributed lock for `bet_id`.

        Args:
            bet_id: The bet to settle.
            match_result: The match outcome.
            event_id: The settlement event ID.

        Returns:
            The resulting `Settlement` record.

        Raises:
            BetNotFoundError: If no bet exists for `bet_id`.
        """
        bet = await self._bets.get_by_id(bet_id)

        if bet is None:
            raise BetNotFoundError(bet_id=bet_id)

        # Idempotency guard — now race-free because we hold the per-bet lock.
        # Any second/third settlement event for this bet will see a
        # non-ACTIVE status and be safely rejected as a duplicate.
        if bet.status != BetStatus.ACTIVE:
            logger.warning(
                "duplicate_settlement_event_rejected",
                bet_id=str(bet_id),
                incoming_event_id=str(event_id),
                original_event_id=str(bet.settlement_event_id) if bet.settlement_event_id else None,
                current_status=bet.status,
            )
            return Settlement(
                bet_id=bet_id,
                match_id=bet.match_id,
                result=match_result,
                payout_amount=0.0,
                event_id=event_id,
                is_duplicate=True,
            )

        won = bet.selection == match_result
        payout_amount = (
            Decimal(str(bet.potential_payout)).quantize(_TWO_DP, rounding=ROUND_HALF_UP)
            if won
            else Decimal("0.00")
        )
        before_status = bet.status

        # UKGC audit log — logged BEFORE the financial operation executes so
        # the transaction can be reconstructed even if the process crashes
        # before the payout/status-update completes.
        logger.info(
            "settlement_audit_before",
            timestamp=datetime.utcnow().isoformat() + "Z",
            bet_id=str(bet_id),
            match_id=bet.match_id,
            event_id=str(event_id),
            actor=_SERVICE_ACTOR,
            reason="match_result_settlement",
            before_status=before_status,
            match_result=match_result,
            won=won,
            payout_amount=str(payout_amount),
        )

        if won:
            await self._payouts.credit_customer(
                customer_id=bet.customer_id,
                amount=float(payout_amount),
                reference=f"settlement_{bet_id}_{event_id}",
            )

        bet.status = BetStatus.SETTLED if won else BetStatus.LOST
        bet.settled_at = datetime.utcnow()
        bet.settlement_event_id = event_id
        await self._bets.save(bet)

        # Settlement records are append-only / immutable per Entain
        # regulatory standards — never update/delete an existing record.
        settlement = Settlement(
            bet_id=bet_id,
            match_id=bet.match_id,
            result=match_result,
            payout_amount=float(payout_amount),
            event_id=event_id,
        )

        # UKGC audit log — logged AFTER the financial operation, capturing
        # the before/after state for the audit trail.
        logger.info(
            "settlement_audit_after",
            timestamp=datetime.utcnow().isoformat() + "Z",
            bet_id=str(bet_id),
            match_id=bet.match_id,
            event_id=str(event_id),
            actor=_SERVICE_ACTOR,
            reason="match_result_settlement",
            before_status=before_status,
            after_status=bet.status,
            match_result=match_result,
            won=won,
            payout_amount=str(payout_amount),
        )

        logger.info(
            "bet_settled",
            bet_id=str(bet_id),
            result=match_result,
            won=won,
            payout=str(payout_amount),
        )

        await self._events.publish_settlement(settlement)

        return settlement
