"""Position tracking for a single tradable asset."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Position:
    """Long/short inventory for one asset.

    Attributes
    ----------
    asset_id:
        Identifier for the instrument (future multi-asset support).
    quantity:
        Signed inventory; positive = long, negative = short.
    average_entry_price:
        Volume-weighted average entry price of the open position.
        Zero when flat.
    """

    asset_id: str = "ASSET"
    quantity: float = 0.0
    average_entry_price: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0.0

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealized_pnl(self, price: float) -> float:
        if self.is_flat:
            return 0.0
        return self.quantity * (price - self.average_entry_price)

    def apply_fill(self, fill_quantity: float, fill_price: float) -> float:
        """Apply a fill and return realized P&L from any closed inventory.

        Increasing (or opening) a position updates the average entry price.
        Reducing (or flipping) realizes P&L against the average entry.
        """
        if fill_quantity == 0.0:
            return 0.0
        if fill_price <= 0.0:
            raise ValueError(f"fill_price must be positive, got {fill_price}")

        realized = 0.0
        q = self.quantity
        new_q = q + fill_quantity

        # Opening from flat or adding to an existing position in the same direction.
        if q == 0.0 or (q > 0.0 and fill_quantity > 0.0) or (q < 0.0 and fill_quantity < 0.0):
            total_cost = self.average_entry_price * abs(q) + fill_price * abs(fill_quantity)
            self.quantity = new_q
            self.average_entry_price = total_cost / abs(new_q) if new_q != 0.0 else 0.0
            return 0.0

        # Closing / reducing / flipping.
        closed = min(abs(fill_quantity), abs(q))
        direction = 1.0 if q > 0.0 else -1.0
        realized = closed * direction * (fill_price - self.average_entry_price)

        if abs(fill_quantity) < abs(q):
            # Partial close: average entry unchanged.
            self.quantity = new_q
        elif abs(fill_quantity) == abs(q):
            # Exact flat.
            self.quantity = 0.0
            self.average_entry_price = 0.0
        else:
            # Flip: remainder opens a new position at fill_price.
            self.quantity = new_q
            self.average_entry_price = fill_price

        return realized
