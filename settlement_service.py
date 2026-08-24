"""Entain Bet Settlement Service.

Fixes the race condition reported in SCRUM-2: when two settlement events
arrive for the same bet within a short window (e.g. a "full time" event
immediately followed by a VAR correction event, < 500ms apart), the previous
implementation would let both calls read status="active" and both process a
payout, resulting in duplicate customer payouts (£14,200 lost over 30 days).

The fix:
  * Wraps the read -> check -> payout -> update critical section in a
    DynamoDB-backed distributed lock, keyed per bet (see `distributed_lock.py`).
  * Re-reads the bet *after* acquiring the lock and re-checks its status, so
    the idempotency check is race-free (only the first attempt for a given
    bet can ever proceed to pay out).
  * Retries lock acquisition with exponential backoff (max 3 attempts, per
    Entain concurrency standards) before failing the settlement attempt.
  * Adds UKGC-compliant structured audit logging before and after the
    financial operation, including before/after bet status and payout
    amount, actor, and correlation ID.
  * Always releases the lock (via `finally`), with the lock TTL acting as a
    safety net if a process crashes while holding it.
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import structlog

from .config import config
from .distributed_lock import DistributedLockRepository
from .exceptions import BetNotFoundError, LockAcquisitionError, PayoutProcessingError
from .models import BetStatus, Settlement

logger = structlog.get_logger()


class BetSettlementService:
    """Settles bets based on match results.

    Uses a distributed lock (DynamoDB conditional writes) to guarantee
    at-most-once payout processing per bet, even when multiple settlement
    events for the same bet arrive concurrently.
    """

    def __init__(
        self,
        bet_repository: Any,
        payout_service: Any,
        event_bus: Any,
        lock_repository: DistributedLockRepository,
    ) -> None:
        """Initialise the settlement service.

        Args:
            bet_repository: Repository for reading/persisting `Bet` entities.
            payout_service: Service used to credit customer balances.
            event_bus: Event bus used to publish settlement events.
            lock_repository: Distributed lock repository (DynamoDB-backed)
                used to serialise settlement of a given bet across
                concurrent callers/processes.
        """
        self._bets = bet_repository
        self._payouts = payout_service
        self._events = event_bus
        self._locks = lock_repository

    async def settle_bet(self, bet_id: UUID, match_result: str, event_id: UUID) -> Settlement:
        """Settle a bet based on the match result.

        Guarantees at-most-once payout processing per bet, even when called
        concurrently (e.g. once from a full-time whistle event and again
        shortly after from a VAR correction event for the same bet).

        Args:
            bet_id: The bet to settle.
            match_result: The match outcome (e.g. "home_win").
            event_id: The settlement event ID that triggered this call.

        Returns:
            The `Settlement` record created, or an idempotent duplicate
            marker (`Settlement.is_duplicate=True`) if the bet had already
            been settled by a prior call.

        Raises:
            BetNotFoundError: If no bet exists for `bet_id`.
            LockAcquisitionError: If the distributed lock cannot be acquired
                after the configured number of retries.
            PayoutProcessingError: If the payout service fails unexpectedly.
        """
        correlation_id = str(uuid4())
        lock_resource_id = f"bet-settlement:{bet_id}"
        holder_id = f"{correlation_id}"
        log = logger.bind(
            correlation_id=correlation_id,
            bet_id=str(bet_id),
            event_id=str(event_id),
            match_result=match_result,
        )

        # UKGC audit log: record intent BEFORE any financial action is taken
        # so the attempt can be reconstructed if the process crashes mid-way.
        log.info(
            "settlement_attempt_started",
            actor="system:bet-settlement-service",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        await self._acquire_lock_with_retry(bet_id, lock_resource_id, holder_id, log)

        try:
            settlement = await self._settle_bet_locked(bet_id, match_result, event_id, log)
        finally:
            await self._release_lock_safely(lock_resource_id, holder_id, log)

        return settlement

    async def _settle_bet_locked(
        self,
        bet_id: UUID,
        match_result: str,
        event_id: UUID,
        log: Any,
    ) -> Settlement:
        """Perform the read -> check -> payout -> update critical section.

        This method must only ever be invoked while holding the distributed
        lock for `bet_id` — it is not safe to call directly.

        Args:
            bet_id: The bet to settle.
            match_result: The match outcome (e.g. "home_win").
            event_id: The settlement event ID that triggered this call.
            log: A structlog bound logger pre-populated with correlation context.

        Returns:
            The resulting `Settlement` record.
        """
        # Authoritative read: this happens while holding the lock, so no
        # other caller can be mid-way through settling this bet concurrently.
        bet = await self._bets.get_by_id(bet_id)
        if bet is None:
            raise BetNotFoundError(bet_id)

        # Idempotency check — now race-free because we hold the lock. If the
        # bet is not ACTIVE, it has already been settled (by this exact event
        # being retried, or by an earlier/concurrent event), so we must not
        # pay out again.
        if bet.status != BetStatus.ACTIVE:
            log.info(
                "duplicate_settlement_prevented",
                bet_status=bet.status,
                previous_settlement_event_id=(
                    str(bet.settlement_event_id) if bet.settlement_event_id else None
                ),
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
        # Round monetary values to 2dp (GBP) per Entain financial standards.
        payout_amount = round(bet.potential_payout, 2) if won else 0.0
        previous_status = bet.status

        # UKGC audit log: log the financial operation BEFORE execution.
        log.info(
            "settlement_payout_before",
            actor="system:bet-settlement-service",
            timestamp=datetime.now(timezone.utc).isoformat(),
            before_status=previous_status,
            won=won,
            payout_amount=payout_amount,
        )

        if won:
            try:
                await self._payouts.credit_customer(
                    customer_id=bet.customer_id,
                    amount=payout_amount,
                    reference=f"settlement_{bet_id}",
                )
            except Exception as exc:  # noqa: BLE001 - wrap all external payout errors
                log.error("settlement_payout_failed", error=str(exc))
                raise PayoutProcessingError(bet_id, str(exc)) from exc

        bet.status = BetStatus.SETTLED if won else BetStatus.LOST
        bet.settled_at = datetime.now(timezone.utc)
        bet.settlement_event_id = event_id
        await self._bets.save(bet)

        # Settlement records are append-only / immutable per UKGC standards —
        # we only ever create new records, never update or delete them.
        settlement = Settlement(
            bet_id=bet_id,
            match_id=bet.match_id,
            result=match_result,
            payout_amount=payout_amount,
            event_id=event_id,
        )

        # UKGC audit log: record the outcome AFTER execution, with a clear
        # before/after trail for regulatory reconstruction.
        log.info(
            "settlement_completed",
            actor="system:bet-settlement-service",
            timestamp=datetime.now(timezone.utc).isoformat(),
            before_status=previous_status,
            after_status=bet.status,
            won=won,
            payout_amount=payout_amount,
        )

        await self._events.publish_settlement(settlement)

        return settlement

    async def _acquire_lock_with_retry(
        self,
        bet_id: UUID,
        resource_id: str,
        holder_id: str,
        log: Any,
    ) -> None:
        """Acquire the distributed lock, retrying with exponential backoff.

        Args:
            bet_id: The bet being settled (used for error reporting only).
            resource_id: The lock resource key (e.g. "bet-settlement:{bet_id}").
            holder_id: Unique identifier for this attempt's lock ownership.
            log: A structlog bound logger pre-populated with correlation context.

        Raises:
            LockAcquisitionError: If the lock could not be acquired within
                `config.max_settlement_retries` attempts.
        """
        max_attempts = config.max_settlement_retries
        ttl_seconds = config.settlement_lock_ttl

        for attempt in range(1, max_attempts + 1):
            acquired = await self._locks.acquire(resource_id, holder_id, ttl_seconds)
            if acquired:
                log.info("settlement_lock_acquired", attempt=attempt)
                return

            log.warning("settlement_lock_contended", attempt=attempt, max_attempts=max_attempts)
            if attempt < max_attempts:
                backoff_seconds = (2 ** (attempt - 1)) + random.uniform(0, 0.25)
                await asyncio.sleep(backoff_seconds)

        log.error("settlement_lock_acquisition_failed", attempts=max_attempts)
        raise LockAcquisitionError(bet_id=bet_id, attempts=max_attempts)

    async def _release_lock_safely(self, resource_id: str, holder_id: str, log: Any) -> None:
        """Release the distributed lock without masking the settlement result.

        A release failure is logged but never raised — the lock's TTL acts
        as a safety net, so a failed release cannot cause a permanent
        deadlock for future settlement attempts.

        Args:
            resource_id: The lock resource key that was acquired.
            holder_id: The holder ID used when the lock was acquired.
            log: A structlog bound logger pre-populated with correlation context.
        """
        try:
            await self._locks.release(resource_id, holder_id)
            log.info("settlement_lock_released")
        except Exception as exc:  # noqa: BLE001
            log.warning("settlement_lock_release_failed", error=str(exc))
