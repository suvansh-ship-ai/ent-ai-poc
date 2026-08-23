"""
Entain Bet Settlement Service — THE BUGGY VERSION
==================================================
This file contains the race condition bug that DEMO-3 asks the AI agent to fix.

BUG: No distributed locking between read and update of bet status.
     When two settlement events arrive concurrently for the same bet,
     both see status="active" and both process the payout.

IMPACT: £14,200 in duplicate payouts over the last 30 days.
"""

import structlog
from uuid import UUID, uuid4
from datetime import datetime

from .models import Bet, BetStatus, Settlement

logger = structlog.get_logger()


class BetSettlementService:
    """Settles bets based on match results.
    
    WARNING: This implementation has a known race condition.
    See DEMO-3 for the fix requirements.
    """

    def __init__(self, bet_repository, payout_service, event_bus):
        self._bets = bet_repository
        self._payouts = payout_service
        self._events = event_bus

    async def settle_bet(self, bet_id: UUID, match_result: str, event_id: UUID) -> Settlement:
        """Settle a bet based on the match result.
        
        BUG: No locking here! Two concurrent calls both read status="active"
        and both process the payout before either updates the status.
        
        Args:
            bet_id: The bet to settle.
            match_result: The match outcome (e.g., "home_win").
            event_id: The settlement event ID.
            
        Returns:
            Settlement record.
        """
        # Step 1: Read the bet
        # BUG: No lock acquired before reading!
        bet = await self._bets.get_by_id(bet_id)
        
        if bet is None:
            raise ValueError(f"Bet {bet_id} not found")
        
        # Step 2: Check if already settled
        # BUG: Between this check and the update below, another thread
        # can also pass this check!
        if bet.status != BetStatus.ACTIVE:
            logger.info("bet_already_settled", bet_id=str(bet_id), status=bet.status)
            return Settlement(bet_id=bet_id, is_duplicate=True)
        
        # Step 3: Determine outcome
        won = (bet.selection == match_result)
        payout_amount = bet.potential_payout if won else 0.0
        
        # Step 4: Process payout (this is the expensive operation)
        if won:
            await self._payouts.credit_customer(
                customer_id=bet.customer_id,
                amount=payout_amount,
                reference=f"settlement_{bet_id}"
            )
        
        # Step 5: Update bet status
        # BUG: By the time we get here, another thread may have already
        # paid out and is about to update status too!
        bet.status = BetStatus.SETTLED if won else BetStatus.LOST
        bet.settled_at = datetime.utcnow()
        bet.settlement_event_id = event_id
        await self._bets.save(bet)
        
        # Step 6: Record settlement
        settlement = Settlement(
            bet_id=bet_id,
            match_id=bet.match_id,
            result=match_result,
            payout_amount=payout_amount,
            event_id=event_id,
        )
        
        logger.info(
            "bet_settled",
            bet_id=str(bet_id),
            result=match_result,
            won=won,
            payout=payout_amount,
        )
        
        # Step 7: Publish event
        await self._events.publish_settlement(settlement)
        
        return settlement
