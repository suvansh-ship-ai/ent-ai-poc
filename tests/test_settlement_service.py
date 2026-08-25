"""Tests for `BetSettlementService` — SCRUM-2 duplicate payout race condition fix.

Covers:
    - Concurrent settlement events for the same bet result in exactly one payout.
    - A duplicate event arriving after settlement is a no-op (idempotency).
    - Lock acquisition failure surfaces as `LockAcquisitionError`.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from distributed_lock import DynamoDBLockClient
from exceptions import LockAcquisitionError
from models import Bet, BetStatus
from settlement_service import BetSettlementService


class FakeInMemoryLockClient(DynamoDBLockClient):
    """In-memory stand-in for `DynamoDBLockClient` used to simulate real
    mutual-exclusion semantics in tests without requiring AWS/DynamoDB.
    """

    def __init__(self) -> None:
        self._locks: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def try_acquire(self, lock_key: str, lock_holder: str, ttl_seconds: int) -> bool:
        async with self._lock:
            if lock_key in self._locks:
                return False
            self._locks[lock_key] = lock_holder
            return True

    async def release(self, lock_key: str, lock_holder: str) -> None:
        async with self._lock:
            if self._locks.get(lock_key) == lock_holder:
                del self._locks[lock_key]


class FakeBetRepository:
    """In-memory bet repository for tests."""

    def __init__(self, bet: Bet) -> None:
        self._bet = bet
        self.save_calls: list[BetStatus] = []

    async def get_by_id(self, bet_id):
        # Return a fresh copy-like read reflecting current in-memory state.
        return self._bet if self._bet.id == bet_id else None

    async def save(self, bet: Bet) -> Bet:
        self._bet = bet
        self.save_calls.append(bet.status)
        return bet


@pytest.fixture
def bet() -> Bet:
    return Bet(
        id=uuid4(),
        customer_id=uuid4(),
        match_id="match-123",
        selection="home_win",
        odds_at_placement=2.0,
        stake=10.0,
        potential_payout=20.0,
        status=BetStatus.ACTIVE,
        placed_at=datetime.utcnow(),
    )


@pytest.fixture
def payout_service() -> AsyncMock:
    service = AsyncMock()
    service.credit_customer = AsyncMock()
    return service


@pytest.fixture
def event_bus() -> AsyncMock:
    bus = AsyncMock()
    bus.publish_settlement = AsyncMock()
    return bus


@pytest.mark.asyncio
async def test_settle_bet_concurrent_events_only_one_payout(bet, payout_service, event_bus):
    """Verify that duplicate settlement events (concurrent) don't cause double payouts."""
    repo = FakeBetRepository(bet)
    lock_client = FakeInMemoryLockClient()
    service = BetSettlementService(repo, payout_service, event_bus, lock_client=lock_client)

    event_a = uuid4()
    event_b = uuid4()

    results = await asyncio.gather(
        service.settle_bet(bet.id, "home_win", event_a),
        service.settle_bet(bet.id, "home_win", event_b),
    )

    duplicates = [r for r in results if r.is_duplicate]
    real_settlements = [r for r in results if not r.is_duplicate]

    assert len(real_settlements) == 1
    assert len(duplicates) == 1
    assert payout_service.credit_customer.call_count == 1
    assert repo._bet.status == BetStatus.SETTLED


@pytest.mark.asyncio
async def test_settle_bet_duplicate_event_after_settlement_is_noop(bet, payout_service, event_bus):
    """Verify that a duplicate event arriving after settlement completes triggers no payout."""
    repo = FakeBetRepository(bet)
    lock_client = FakeInMemoryLockClient()
    service = BetSettlementService(repo, payout_service, event_bus, lock_client=lock_client)

    first_event = uuid4()
    await service.settle_bet(bet.id, "home_win", first_event)
    assert payout_service.credit_customer.call_count == 1

    duplicate_event = uuid4()
    result = await service.settle_bet(bet.id, "home_win", duplicate_event)

    assert result.is_duplicate is True
    assert result.payout_amount == 0.0
    assert payout_service.credit_customer.call_count == 1


@pytest.mark.asyncio
async def test_settle_bet_lock_never_acquired_raises_lock_acquisition_error(bet, payout_service, event_bus):
    """Verify that repeated lock acquisition failure surfaces as LockAcquisitionError."""
    repo = FakeBetRepository(bet)

    class AlwaysBusyLockClient(DynamoDBLockClient):
        def __init__(self) -> None:
            pass

        async def try_acquire(self, lock_key: str, lock_holder: str, ttl_seconds: int) -> bool:
            return False

        async def release(self, lock_key: str, lock_holder: str) -> None:
            return None

    service = BetSettlementService(repo, payout_service, event_bus, lock_client=AlwaysBusyLockClient())

    with pytest.raises(LockAcquisitionError):
        await service.settle_bet(bet.id, "home_win", uuid4())

    assert payout_service.credit_customer.call_count == 0
