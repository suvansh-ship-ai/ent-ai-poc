"""Entain Sports Betting Platform — Distributed Locking (DynamoDB).

Provides a distributed lock implementation backed by DynamoDB conditional
writes. This is used to serialise critical financial operations (e.g. bet
settlement) across multiple concurrent workers/processes/requests, which
prevents race conditions such as the duplicate-payout bug described in
SCRUM-2.

Per Entain engineering standards, DynamoDB (not Redis) is the standardised
backend for distributed locks. Locks carry a TTL so a crashed holder can
never deadlock other callers indefinitely.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import structlog
from botocore.exceptions import ClientError

from .config import config
from .exceptions import DistributedLockError

logger = structlog.get_logger()


class DistributedLockRepository:
    """Data-access layer for acquiring/releasing distributed locks in DynamoDB.

    Uses a conditional write so that only one caller can hold the lock for a
    given resource at a time:

        ConditionExpression = "attribute_not_exists(resource_id) OR expires_at < :now"

    Lock records include an `expires_at` TTL attribute so that a holder that
    crashes before releasing the lock cannot deadlock future callers.
    """

    def __init__(self, dynamodb_client: Any, table_name: str | None = None) -> None:
        """Initialise the lock repository.

        Args:
            dynamodb_client: An async DynamoDB client exposing `put_item` and
                `delete_item` methods compatible with the boto3/aioboto3 API
                (low-level `TableName`/`Item`/`Key` style calls).
            table_name: Override for the DynamoDB table name. Defaults to
                `config.dynamodb_table_locks`.
        """
        self._client = dynamodb_client
        self._table_name = table_name or config.dynamodb_table_locks

    async def acquire(self, resource_id: str, holder_id: str, ttl_seconds: int) -> bool:
        """Attempt to acquire a lock for `resource_id`.

        Args:
            resource_id: Unique identifier of the resource being locked
                (e.g. ``f"bet-settlement:{bet_id}"``).
            holder_id: Unique identifier for this attempt's lock ownership
                (e.g. a correlation ID). Used defensively on release.
            ttl_seconds: How long the lock is valid for before it is
                considered expired and eligible to be taken over.

        Returns:
            True if the lock was acquired, False if it is currently held by
            another caller and has not yet expired.

        Raises:
            DistributedLockError: If the underlying DynamoDB call fails for a
                reason other than lock contention.
        """
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat()

        try:
            await self._client.put_item(
                TableName=self._table_name,
                Item={
                    "resource_id": {"S": resource_id},
                    "lock_holder": {"S": holder_id},
                    "acquired_at": {"S": now.isoformat()},
                    "expires_at": {"S": expires_at},
                },
                ConditionExpression="attribute_not_exists(resource_id) OR expires_at < :now",
                ExpressionAttributeValues={":now": {"S": now.isoformat()}},
            )
            return True
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code == "ConditionalCheckFailedException":
                # Expected outcome when the lock is currently held — not an error.
                return False
            logger.error("distributed_lock_acquire_failed", resource_id=resource_id, error=str(exc))
            raise DistributedLockError(resource_id, str(exc)) from exc

    async def release(self, resource_id: str, holder_id: str) -> None:
        """Release a previously acquired lock.

        Args:
            resource_id: Unique identifier of the locked resource.
            holder_id: The holder that originally acquired the lock. Used as
                a defensive check so a caller can never release a lock it
                does not currently own (e.g. after TTL expiry + takeover).

        Raises:
            DistributedLockError: If the underlying DynamoDB call fails for a
                reason other than "not the current holder".
        """
        try:
            await self._client.delete_item(
                TableName=self._table_name,
                Key={"resource_id": {"S": resource_id}},
                ConditionExpression="lock_holder = :holder",
                ExpressionAttributeValues={":holder": {"S": holder_id}},
            )
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code == "ConditionalCheckFailedException":
                # Lock already expired and was taken over by another holder,
                # or was never acquired by us. Safe to no-op.
                logger.warning(
                    "distributed_lock_release_skipped_not_holder",
                    resource_id=resource_id,
                    holder_id=holder_id,
                )
                return
            logger.error("distributed_lock_release_failed", resource_id=resource_id, error=str(exc))
            raise DistributedLockError(resource_id, str(exc)) from exc
