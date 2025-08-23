# portfolio.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
from collections import defaultdict

from trader.events import MarketEvent, SignalEvent, OrderEvent, FillEvent, Event
from trader.config import Settings
from utilts.logs import logs


@dataclass
class Position:
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0
    total_buy_commission: float = 0.0
    total_buy_quantity: int = 0

    def update_on_fill(self, fill: FillEvent) -> None:
        if fill.direction == "BUY":
            total_cost = self.avg_price * self.quantity
            fill_cost = fill.price * fill.quantity
            new_total_qty = self.quantity + fill.quantity
            self.avg_price = ((total_cost + fill_cost) / new_total_qty) if new_total_qty > 0 else 0.0
            self.quantity = new_total_qty
            self.total_buy_commission += fill.commission
            self.total_buy_quantity += fill.quantity
        elif fill.direction == "SELL":
            self.quantity -= fill.quantity
            if self.quantity == 0:
                self.avg_price = 0.0
            if self.total_buy_quantity > 0:
                proportion = fill.quantity / self.total_buy_quantity
                self.total_buy_commission *= (1 - proportion)
                self.total_buy_quantity -= fill.quantity


@dataclass
class TradeRecord:
    symbol: str
    date: datetime
    direction: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    commission: float
    stamp_tax: float
    realized_pnl: float


@dataclass
class DailySnapshot:
    date: datetime
    cash: float
    equity: float
    holdings: Dict[str, float]
    total_value: float


class Portfolio:
    """Tracks positions, cash, equity, and transaction history."""

    MIN_COMMISSION: float = 5.0
    COMMISSION_RATE: float = 0.001
    STAMP_DUTY_RATE: float = 0.001
    TRANSFER_FEE_RATE: float = 0.00001

    def __init__(self, settings: Settings):
        self.settings = settings
        self.cash: float = settings.trading.INITIAL_CASH
        self.risk_pct: float = settings.trading.RISK_PCT

        self.positions: Dict[str, Position] = {}
        self.current_prices: Dict[str, float] = {}
        self.transactions: List[TradeRecord] = []
        self.daily_snapshots: List[DailySnapshot] = []
        self.realized_pnl: Dict[str, float] = {}
        self.symbol_equity_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.history: List[Tuple[datetime, float]] = []
        self.cash_records = []

    # -----------------------------
    # Properties
    # -----------------------------
    @property
    def equity(self) -> float:
        total = self.cash + sum(
            pos.quantity * self.current_prices.get(sym, 0.0) for sym, pos in self.positions.items()
        )
        return total

    @property
    def equity_curve(self) -> List[Tuple[datetime, float]]:
        return self.history

    @property
    def cash_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.cash_records, columns=["date", "cash"]).groupby('date').sum()

    @property
    def equity_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.history, columns=["datetime", "equity"]).groupby("datetime").sum()
        return df
        # assume same weight(quantity)

    @property
    def symbol_equity_df(self) -> pd.DataFrame:
        frames = []
        for symbol, history in self.symbol_equity_history.items():
            df = pd.DataFrame(history, columns=["datetime", symbol]).set_index("datetime")
            df = df[df[symbol] != 0].sort_index()
            frames.append(df)

        return pd.concat(frames)

    # -----------------------------
    # Price & Signal Handling
    # -----------------------------
    def update_price(self, market_event: MarketEvent) -> None:
        self.current_prices[market_event.symbol] = market_event.close

    def on_signal(self, signal: SignalEvent) -> Optional[OrderEvent]:

        symbol = signal.symbol
        direction = signal.signal_type
        price = self.current_prices.get(symbol, 0.0)
        quantity = 10  # TODO: could calculate based on risk_pct

        skip_event = Event(None, None)  # hold

        if direction == "BUY" and self.cash < price * quantity:
            logs.record_log(f"Not enough cash for {symbol} BUY signal", level=3)
            return
        if direction == "SELL" and (symbol not in self.positions or self.positions[symbol].quantity < quantity):
            logs.record_log(f"Not enough holdings for {symbol} SELL signal", level=3)
            return

        return OrderEvent(symbol, "MKT", quantity, direction, signal.datetime)

    # -----------------------------
    # Fill Handling
    # -----------------------------
    def on_fill(self, fill: FillEvent) -> None:
        symbol = fill.symbol
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)

        position = self.positions[symbol]

        if fill.direction == "BUY":
            commission = self._calculate_buy_commission(fill.price * fill.quantity)
            self.cash -= fill.price * fill.quantity + commission
            fill.commission = commission

        elif fill.direction == "SELL":
            commission = self._calculate_sell_commission(fill.price * fill.quantity)
            if position.total_buy_quantity == 0:
                logs.record_log(f"Attempted to sell without holdings: {symbol}", level=3)
                return
            buy_fee_applied = (fill.quantity / position.total_buy_quantity) * position.total_buy_commission
            gross_pnl = (fill.price - position.avg_price) * fill.quantity
            realized = gross_pnl - buy_fee_applied - commission
            self.realized_pnl[symbol] = self.realized_pnl.get(symbol, 0.0) + realized
            self.cash += fill.price * fill.quantity - commission
            fill.commission = commission
        else:
            return

        logs.record_log(f"succeed to fill {fill}")

        position.update_on_fill(fill)

        self.transactions.append(
            TradeRecord(
                symbol=symbol,
                date=fill.datetime,
                direction=fill.direction,
                quantity=fill.quantity,
                price=fill.price,
                commission=fill.commission,
                stamp_tax=self.STAMP_DUTY_RATE if fill.direction == "SELL" else 0.0,
                realized_pnl=self.realized_pnl.get(symbol, 0.0)
            )
        )

    # -----------------------------
    # Risk & Commission Utilities
    # -----------------------------
    def calculate_quantity(self, price: float) -> int:
        qty = int((self.cash * self.risk_pct) // price)
        return max(qty, 1)

    def _calculate_buy_commission(self, amount: float) -> float:
        return max(amount * self.COMMISSION_RATE, self.MIN_COMMISSION)

    def _calculate_sell_commission(self, amount: float) -> float:
        commission = max(amount * self.COMMISSION_RATE, self.MIN_COMMISSION)
        stamp = amount * self.STAMP_DUTY_RATE
        transfer = amount * self.TRANSFER_FEE_RATE
        return commission + stamp + transfer

    # -----------------------------
    # Daily Snapshot
    # -----------------------------
    def record_daily_snapshot(self, date: datetime) -> None:
        holdings_value = {
            symbol: pos.quantity * self.current_prices.get(symbol, 0.0)
            for symbol, pos in self.positions.items()
        }
        total_equity = sum(holdings_value.values()) + self.cash
        self.cash_records.append((date, self.cash))

        self.history.append((date, total_equity))  # account equity(symbol value+cash)
        self.daily_snapshots.append(
            DailySnapshot(
                date=date,
                cash=self.cash,
                equity=total_equity,
                holdings=holdings_value,
                total_value=total_equity
            )
        )

        for symbol, value in holdings_value.items():
            if symbol not in self.symbol_equity_history:
                self.symbol_equity_history[symbol] = []  # symbol equity
            if value:
                self.symbol_equity_history[symbol].append((date, value))