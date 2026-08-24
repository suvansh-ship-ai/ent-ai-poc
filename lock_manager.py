"""Entain Sports Betting Platform — Distributed Lock Manager.

Provides DynamoDB-backed distributed locking for settlement, payout, and
balance operations, per Entain's concurrency & distributed state standards:

    - DynamoDB conditional writes are used as the locking primitive
      (NOT Redis — Entain standardises on DynamoDB for distributed locks).
    - Lock pattern: ConditionExpression =
          "attribute_not_exists(lock_holder) OR expires_at < :now"
    - Lock TTL defaults to 30 seconds (prevents deadlocks if a holder
      crashes while holding the lock).
    - Acquisition retries with exponential backoff, max 3 attempts by
      default.

This module is intentionally scoped to bet-level locks (used by
BetSettlementService) so that lock failures can be surfaced using the
existing `LockAcquisitionError` exception, which is keyed by bet_id.
"""

import asyncio
import random
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import structlog
from botocore.exceptions import ClientError

from .config import config
from .exceptions import LockAcquisitionError

logger = structlog.get_logger()

_CONDITIONAL_CHECK_FAILED = "ConditionalCheckFailedException"


class DynamoDBLockManager:
    """Acquires and releases per-bet distributed locks backed by DynamoDB.

    The lock record schema (single-table) is:
        lock_key    (S, partition key) — e.g. "bet:{bet_id}"
        lock_holder (S)                — opaque identifier of the current holder
        acquired_at (S, ISO 8601 UTC)
        expires_at  (S, ISO 8601 UTC)  — used to reclaim locks from crashed holders
    """

    def __init__(
        self,
        dynamodb_client: Any,
        table_name: str | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        """Initialize the lock manager.

        Args:
            dynamodb_client: An async DynamoDB client exposing `put_item` and
                `delete_item` (e.g. an aioboto3 DynamoDB client).
            table_name: DynamoDB table used to store lock records. Defaults
                to `config.dynamodb_table_locks`.
            ttl_seconds: Lock time-to-live in seconds. Defaults to
                `config.lock_ttl_seconds`.
        """
        self._client = dynamodb_client
        self._table_name = table_name or config.dynamodb_table_locks
        self._ttl_seconds = ttl_seconds or config.lock_ttl_seconds

    @staticmethod
    def _lock_key(bet_id: UUID) -> str:
        """Build the DynamoDB partition key for a bet's settlement lock."""
        return f"bet:{bet_id}"

    async def acquire_bet_lock(
        self,
        bet_id: UUID,
        holder: str,
        max_attempts: int | None = None,
    ) -> None:
        """Acquire a distributed lock for a bet, retrying with backoff.

        Args:
            bet_id: The bet whose settlement critical section is being locked.
            holder: Opaque identifier of the caller (e.g. "settlement:{event_id}"),
                used for audit trail purposes.
            max_attempts: Maximum number of acquisition attempts. Defaults to
                `config.max_settlement_retries`.

        Raises:
            LockAcquisitionError: If the lock cannot be acquired after
                `max_attempts` attempts.
        """
        attempts = max_attempts or config.max_settlement_retries
        lock_key = self._lock_key(bet_id)

        for attempt in range(1, attempts + 1):
            now = datetime.utcnow()
            expires_at = now + timedelta(seconds=self._ttl_seconds)

            try:
                await self._client.put_item(
                    TableName=self._table_name,
                    Item={
                        "lock_key": {"S": lock_key},
                        "lock_holder": {"S": holder},
                        "acquired_at": {"S": now.isoformat()},
                        "expires_at": {"S": expires_at.isoformat()},
                    },
                    ConditionExpression=(
                        "attribute_not_exists(lock_holder) OR expires_at < :now"
                    ),
                    ExpressionAttributeValues={":now": {"S": now.isoformat()}},
                )
                logger.info(
                    "settlement_lock_acquired",
                    bet_id=str(bet_id),
                    holder=holder,
                    attempt=attempt,
                    ttl_seconds=self._ttl_seconds,
                )
                return
            except ClientError as exc:
                error_code = exc.response.get("Error", {}).get("Code", "")
                if error_code != _CONDITIONAL_CHECK_FAILED:
                    logger.error(
                        "settlement_lock_unexpected_error",
                        bet_id=str(bet_id),
                        holder=holder,
                        attempt=attempt,
                        error=str(exc),
                    )
                    raise

                if attempt >= attempts:
                    logger.warning(
                        "settlement_lock_acquisition_exhausted",
                        bet_id=str(bet_id),
                        holder=holder,
                        attempts=attempts,
                    )
                    raise LockAcquisitionError(bet_id=bet_id, attempts=attempts) from exc

                backoff_seconds = (2 ** (attempt - 1)) + random.uniform(0, 0.25)
                logger.info(
                    "settlement_lock_acquisition_retry",
                    bet_id=str(bet_id),
                    holder=holder,
                    attempt=attempt,
                    backoff_seconds=round(backoff_seconds, 3),
                )
                await asyncio.sleep(backoff_seconds)

    async def release_bet_lock(self, bet_id: UUID, holder: str) -> None:
        """Release a previously acquired lock for a bet.

        Only releases the lock if `holder` is still the current lock holder,
        preventing a slow caller from releasing a lock that has since expired
        and been re-acquired by another process.

        Args:
            bet_id: The bet whose lock should be released.
            holder: The identifier that originally acquired the lock.
        """
        lock_key = self._lock_key(bet_id)

        try:
            await self._client.delete_item(
                TableName=self._table_name,
                Key={"lock_key": {"S": lock_key}},
                ConditionExpression="lock_holder = :holder",
                ExpressionAttributeValues={":holder": {"S": holder}},
            )
            logger.info("settlement_lock_released", bet_id=str(bet_id), holder=holder)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code == _CONDITIONAL_CHECK_FAILED:
                # Lock was already reclaimed (expired) by another holder — safe to ignore.
                logger.warning(
                    "settlement_lock_release_no_longer_owner",
                    bet_id=str(bet_id),
                    holder=holder,
                )
                return
            logger.error(
                "settlement_lock_release_failed",
                bet_id=str(bet_id),
                holder=holder,
                error=str(exc),
            )
            raise
