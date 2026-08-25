"""Entain Bet Settlement Service.

Fixes SCRUM-2: a race condition where concurrent settlement events for the
same bet (e.g., final whistle + VAR correction arriving within ~500ms) could
both observe status="active" and both trigger a payout — resulting in
duplicate customer credits (£14,200 financial impact over 30 days).

Root cause (previous implementation):
    `settle_bet()` read the bet status, processed the payout, and only then
    updated the status — with no lock between read and update. Two concurrent
    calls could both pass the "already settled?" check before either had
    written back the new status.

Fix:
    - A distributed lock (DynamoDB conditional write, per Entain concurrency
      standards) is acquired on the bet BEFORE it is read, and held until the
      settlement record is persisted and the status update is committed. This
      closes the read-check-act race window entirely.
    - The idempotency check is performed *inside* the lock (defence in depth),
      so a duplicate event that arrives after settlement has completed is
      safely ignored rather than triggering a second payout.
    - Every settlement attempt is audit-logged BEFORE and AFTER the financial
      operation executes, including before/after status and payout amount,
      per UKGC regulatory requirements.
    - Lock acquisition uses retry with exponential backoff (max 3 attempts by
      default); if the lock can never be acquired, `LockAcquisitionError` is
      raised so the caller can retry the event later rather than silently
      dropping it.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import structlog

from .config import config
from .distributed_lock import DistributedLock, DynamoDBLockClient
from .exceptions import BetNotFoundError, LockAcquisitionError
from .models import BetStatus, Settlement

logger = structlog.get_logger()


class BetSettlementService:
    """Settles bets based on match results, with idempotency and distributed locking.

    Concurrency safety:
        Settlement is serialized per-bet via a DynamoDB-backed distributed
        lock (see `distributed_lock.DistributedLock`). Only one caller may
        hold the lock for a given `bet_id` at a time, so the
        read -> decide -> payout -> update sequence is atomic with respect to
        any other concurrent settlement attempt for the same bet.
    """

    def __init__(
        self,
        bet_repository,
        payout_service,
        event_bus,
        lock_client: DynamoDBLockClient | None = None,
    ) -> None:
        """Initialise the service with its dependencies.

        Args:
            bet_repository: Repository providing `get_by_id(bet_id)` and `save(bet)`.
            payout_service: Service providing `credit_customer(customer_id, amount, reference)`.
            event_bus: Event bus providing `publish_settlement(settlement)`.
            lock_client: Distributed lock client. Defaults to a new
                `DynamoDBLockClient` using application configuration.
        """
        self._bets = bet_repository
        self._payouts = payout_service
        self._events = event_bus
        self._lock_client = lock_client or DynamoDBLockClient()

    async def settle_bet(self, bet_id: UUID, match_result: str, event_id: UUID) -> Settlement:
        """Settle a bet based on the match result, guarding against concurrent duplicates.

        Args:
            bet_id: The bet to settle.
            match_result: The match outcome (e.g., "home_win").
            event_id: The settlement event that triggered this call. Used for
                idempotency tracking and the audit trail.

        Returns:
            The resulting `Settlement` record. If the bet had already been
            settled by a prior (or concurrently-won) event, a `Settlement`
            with `is_duplicate=True` and `payout_amount=0.0` is returned and
            no further action is taken.

        Raises:
            BetNotFoundError: If no bet exists for `bet_id`.
            LockAcquisitionError: If the distributed lock cannot be acquired
                after retries. The caller should retry the event later.
        """
        logger.info(
            "settlement_attempt_started",
            bet_id=str(bet_id),
            event_id=str(event_id),
            match_result=match_result,
            actor="settlement_engine",
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

        try:
            async with DistributedLock(
                lock_client=self._lock_client,
                bet_id=bet_id,
                holder=str(event_id),
                ttl_seconds=config.settlement_lock_ttl,
                max_attempts=config.max_settlement_retries,
            ):
                return await self._settle_bet_locked(bet_id, match_result, event_id)
        except LockAcquisitionError:
            logger.error(
                "settlement_lock_acquisition_failed",
                bet_id=str(bet_id),
                event_id=str(event_id),
                actor="settlement_engine",
            )
            raise

    async def _settle_bet_locked(self, bet_id: UUID, match_result: str, event_id: UUID) -> Settlement:
        """Perform the settlement while the per-bet distributed lock is held.

        Precondition: caller holds the settlement lock for `bet_id`, so the
        read-check-act sequence below cannot race with another settlement
        attempt for the same bet.

        Args:
            bet_id: The bet to settle.
            match_result: The match outcome (e.g., "home_win").
            event_id: The settlement event that triggered this call.

        Returns:
            The resulting `Settlement` record.

        Raises:
            BetNotFoundError: If no bet exists for `bet_id`.
        """
        # Step 1: Read the bet. Safe — protected by the distributed lock.
        bet = await self._bets.get_by_id(bet_id)
        if bet is None:
            raise BetNotFoundError(bet_id=bet_id)

        # Step 2: Idempotency check, re-verified inside the lock (defence in depth).
        # Any bet not in an unsettled state has already been processed — either
        # by an earlier event or (pre-fix) by a racing duplicate — so we must
        # not pay out again.
        if bet.status not in (BetStatus.ACTIVE, BetStatus.PENDING):
            logger.info(
                "bet_already_settled_duplicate_event_ignored",
                bet_id=str(bet_id),
                event_id=str(event_id),
                existing_status=bet.status,
                existing_settlement_event_id=(
                    str(bet.settlement_event_id) if bet.settlement_event_id else None
                ),
                actor="settlement_engine",
            )
            return Settlement(
                bet_id=bet_id,
                match_id=bet.match_id,
                result=match_result,
                payout_amount=0.0,
                event_id=event_id,
                is_duplicate=True,
            )

        # Step 3: Determine outcome.
        won = bet.selection == match_result
        payout_amount = round(bet.potential_payout, 2) if won else 0.0
        previous_status = bet.status

        # UKGC audit log — BEFORE the financial operation executes.
        logger.info(
            "settlement_payout_audit_before",
            bet_id=str(bet_id),
            customer_id=str(bet.customer_id),
            event_id=str(event_id),
            match_id=bet.match_id,
            match_result=match_result,
            selection=bet.selection,
            won=won,
            payout_amount=payout_amount,
            previous_status=previous_status,
            timestamp=datetime.utcnow().isoformat() + "Z",
            actor="settlement_engine",
        )

        # Step 4: Process payout. Only the lock holder can reach this line for
        # a given bet, eliminating the double-payout race.
        if won:
            try:
                await self._payouts.credit_customer(
                    customer_id=bet.customer_id,
                    amount=payout_amount,
                    reference=f"settlement_{bet_id}_{event_id}",
                )
            except Exception:
                logger.error(
                    "settlement_payout_failed",
                    bet_id=str(bet_id),
                    event_id=str(event_id),
                    payout_amount=payout_amount,
                    actor="settlement_engine",
                )
                raise

        # Step 5: Update bet status.
        bet.status = BetStatus.SETTLED if won else BetStatus.LOST
        bet.settled_at = datetime.utcnow()
        bet.settlement_event_id = event_id
        await self._bets.save(bet)

        # Step 6: Record settlement (append-only / immutable per UKGC retention rules).
        settlement = Settlement(
            bet_id=bet_id,
            match_id=bet.match_id,
            result=match_result,
            payout_amount=payout_amount,
            event_id=event_id,
            is_duplicate=False,
        )

        # UKGC audit log — AFTER the financial operation executes, with before/after values.
        logger.info(
            "settlement_payout_audit_after",
            bet_id=str(bet_id),
            customer_id=str(bet.customer_id),
            event_id=str(event_id),
            match_id=bet.match_id,
            previous_status=previous_status,
            new_status=bet.status,
            payout_amount=payout_amount,
            settled_at=bet.settled_at.isoformat() + "Z",
            actor="settlement_engine",
        )

        # Step 7: Publish settlement event downstream.
        await self._events.publish_settlement(settlement)

        return settlement
