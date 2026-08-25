"""Entain Sports Betting Platform — Distributed Locking (DynamoDB).

Provides a distributed lock primitive backed by DynamoDB conditional writes.
Used to serialize critical financial operations (e.g., bet settlement) across
concurrent service instances/threads, preventing race conditions such as the
duplicate-payout bug described in SCRUM-2.

Standards followed (see Entain Knowledge Base — Concurrency & Distributed State):
    - DynamoDB conditional writes (NOT Redis) for distributed locks.
    - ConditionExpression = "attribute_not_exists(lock_key) OR expires_at < :now"
    - Lock TTL: 30 seconds default (prevents deadlocks on crashed holders).
    - Retry with exponential backoff, max 3 attempts by default.
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta
from types import TracebackType
from uuid import UUID

import structlog

from .config import config
from .exceptions import LockAcquisitionError

logger = structlog.get_logger()


def _is_conditional_check_failed(exc: Exception) -> bool:
    """Return True if `exc` represents a DynamoDB ConditionalCheckFailedException.

    Args:
        exc: The exception raised by boto3/botocore during a put/delete call.

    Returns:
        True if the exception is a conditional check failure (i.e., the lock
        is already held by someone else), False otherwise.
    """
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return response.get("Error", {}).get("Code") == "ConditionalCheckFailedException"
    return False


class DynamoDBLockClient:
    """Wraps a DynamoDB table used to acquire/release distributed locks.

    The backing table is expected to have:
        - Partition key: `lock_key` (str)
        - Attributes: `lock_holder` (str), `acquired_at` (str, ISO 8601),
          `expires_at` (float, epoch seconds)
    """

    def __init__(self, table_name: str | None = None, region: str | None = None) -> None:
        """Initialise the lock client.

        Args:
            table_name: DynamoDB table name. Defaults to `config.dynamodb_table_locks`.
            region: AWS region. Defaults to `config.dynamodb_region`.
        """
        self._table_name = table_name or config.dynamodb_table_locks
        self._region = region or config.dynamodb_region
        self._table = None

    def _get_table(self):
        """Lazily create the boto3 DynamoDB Table resource.

        Lazy initialisation keeps this module importable in unit tests without
        requiring AWS credentials to be configured.
        """
        if self._table is None:
            import boto3  # local import: avoids a hard dependency at import time

            resource = boto3.resource("dynamodb", region_name=self._region)
            self._table = resource.Table(self._table_name)
        return self._table

    async def try_acquire(self, lock_key: str, lock_holder: str, ttl_seconds: int) -> bool:
        """Attempt to acquire the lock via a DynamoDB conditional write.

        Args:
            lock_key: Unique key identifying the resource being locked.
            lock_holder: Identifier of the caller attempting to acquire the lock
                (e.g., the settlement event ID), used for audit/debugging.
            ttl_seconds: Lock validity duration before it is considered expired.

        Returns:
            True if the lock was acquired, False if it is currently held by
            another (non-expired) holder.

        Raises:
            Exception: Re-raises any unexpected (non-conditional-check) error
                from the underlying DynamoDB call.
        """
        table = self._get_table()
        now = datetime.utcnow()
        expires_at = (now + timedelta(seconds=ttl_seconds)).timestamp()

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: table.put_item(
                    Item={
                        "lock_key": lock_key,
                        "lock_holder": lock_holder,
                        "acquired_at": now.isoformat(),
                        "expires_at": expires_at,
                    },
                    ConditionExpression="attribute_not_exists(lock_key) OR expires_at < :now",
                    ExpressionAttributeValues={":now": now.timestamp()},
                ),
            )
            return True
        except Exception as exc:  # noqa: BLE001 - boto3 raises ClientError; handled generically for portability
            if _is_conditional_check_failed(exc):
                return False
            raise

    async def release(self, lock_key: str, lock_holder: str) -> None:
        """Release the lock, only if still held by `lock_holder`.

        Args:
            lock_key: Unique key identifying the resource that was locked.
            lock_holder: Identifier of the caller that acquired the lock.
        """
        table = self._get_table()
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: table.delete_item(
                    Key={"lock_key": lock_key},
                    ConditionExpression="lock_holder = :holder",
                    ExpressionAttributeValues={":holder": lock_holder},
                ),
            )
        except Exception as exc:  # noqa: BLE001
            if _is_conditional_check_failed(exc):
                logger.warning(
                    "lock_release_skipped_not_owner",
                    lock_key=lock_key,
                    lock_holder=lock_holder,
                )
                return
            raise


class DistributedLock:
    """Async context manager for acquiring a per-bet settlement lock.

    Retries with exponential backoff (+ jitter) up to `max_attempts` times
    before raising `LockAcquisitionError`.

    Example:
        async with DistributedLock(lock_client, bet_id=bet_id, holder=str(event_id)):
            ...critical section (read -> decide -> payout -> update)...
    """

    def __init__(
        self,
        lock_client: DynamoDBLockClient,
        bet_id: UUID,
        holder: str,
        ttl_seconds: int | None = None,
        max_attempts: int = 3,
    ) -> None:
        """Initialise the lock context manager.

        Args:
            lock_client: The DynamoDB-backed lock client to use.
            bet_id: The bet being locked. Used to derive the lock key and for
                error reporting.
            holder: Identifier of the current caller (e.g., settlement event ID).
            ttl_seconds: Lock TTL. Defaults to `config.settlement_lock_ttl`.
            max_attempts: Maximum acquisition attempts before giving up.
        """
        self._client = lock_client
        self._bet_id = bet_id
        self._lock_key = f"settlement:{bet_id}"
        self._holder = holder
        self._ttl_seconds = ttl_seconds or config.settlement_lock_ttl
        self._max_attempts = max_attempts

    async def __aenter__(self) -> "DistributedLock":
        """Acquire the lock, retrying with exponential backoff.

        Raises:
            LockAcquisitionError: If the lock could not be acquired within
                `max_attempts` attempts.
        """
        for attempt in range(1, self._max_attempts + 1):
            acquired = await self._client.try_acquire(self._lock_key, self._holder, self._ttl_seconds)
            if acquired:
                logger.info(
                    "lock_acquired",
                    lock_key=self._lock_key,
                    holder=self._holder,
                    attempt=attempt,
                )
                return self

            backoff_seconds = min(0.5 * (2 ** (attempt - 1)), 5.0) + random.uniform(0, 0.1)
            logger.warning(
                "lock_acquisition_retry",
                lock_key=self._lock_key,
                holder=self._holder,
                attempt=attempt,
                backoff_seconds=round(backoff_seconds, 3),
            )
            await asyncio.sleep(backoff_seconds)

        logger.error(
            "lock_acquisition_failed",
            lock_key=self._lock_key,
            holder=self._holder,
            attempts=self._max_attempts,
        )
        raise LockAcquisitionError(bet_id=self._bet_id, attempts=self._max_attempts)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release the lock on exit, regardless of whether an exception occurred."""
        await self._client.release(self._lock_key, self._holder)
        logger.info("lock_released", lock_key=self._lock_key, holder=self._holder)
