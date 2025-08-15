# portfolio.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import pandas as pd

from collections import defaultdict
from trader.events import OrderEvent, FillEvent, EventType, SignalEvent, MarketEvent

from utilts.logs import logs

from trader.config import Settings

from trader.events import FillEvent, SignalEvent
from typing import Dict, List, Tuple


@dataclass
class Position:
    #     """represents the full state of an asset you own"""
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0
    total_buy_commission = 0.0
    total_buy_quantity = 0

    def update_on_fill(self, fill: FillEvent):
        if fill.direction == "BUY":

            total_cost = self.avg_price * self.quantity
            fill_cost = fill.price * fill.quantity
            new_total_qty = self.quantity + fill.quantity
            self.avg_price = (
                (total_cost + fill_cost) / new_total_qty if new_total_qty > 0 else 0.0
            )
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
    holdings: Dict[str, float]  # symbol -> market value
    total_value: float


class Portfolio:
    """
    The Portfolio class tracks cash, positions, and market values over time.
    It records per-symbol market values and total portfolio value at each market update,
    enabling performance analysis and plotting versus a benchmark.
    """
    # Commission and fee rates (China stock market)
    min_commission = 5.0  # Min ¥5
    stamp_duty_rate = 0.001  # 0.1% (sell only)
    transfer_fee_rate = 0.00001  # 0.001% (sell only)
    commission_rate = 0.001  # 0.1%

    def __init__(self, events, settings: Settings):

        self.events = events
        self.settings = settings
        self.cash = settings.trading.INITIAL_CASH  # reflects all costs accurately.
        self.risk_pct = settings.trading.RISK_PCT

        self.positions: Dict[str, Position] = {}  # updates quantity and avg price.
        self.current_prices: Dict[str, float] = {}
        self.history: List[Tuple[datetime, float]] = []

        self.transactions: List[TradeRecord] = []
        self.daily_snapshots: List[DailySnapshot] = []
        self.realized_pnl: Dict[str, float] = {}
        self.symbol_equity_history: Dict[str, List[Tuple[datetime, float]]] = {}  # daily per-symbol value

        # Realized PnL = (Sell Price - Buy Price) × Quantity - (Buy Commission + Sell Commission)
        # Realized PnL = (Sell Price - Buy Price) × Quantity
        #                - Buy Commission - (Sell Commission + Stamp Duty + Transfer Fee) in china

    @property
    def equity_curve(self):
        return self.history

    def update_price(self, market_event: MarketEvent):
        self.current_prices[market_event.symbol] = market_event.close

    # def update_price(self, market_event: MarketEvent):
    #     symbol = market_event.symbol
    #     price = market_event.close
    #     self.current_prices[symbol] = price
    #
    #     # Record account-level equity
    #     total_equity = self.cash + sum(
    #         pos.quantity * self.current_prices.get(pos.symbol, 0)
    #         for pos in self.positions.values()
    #     )
    #     self.history.append((market_event.datetime, total_equity))
    #
    #     # Record per-symbol equity: symbol_value + allocated cash (set to 0 if you want)
    #     pos = self.positions.get(symbol)
    #     symbol_value = pos.quantity * price if pos else 0.0
    #     if symbol not in self.symbol_equity_history:
    #         self.symbol_equity_history[symbol] = []
    #     self.symbol_equity_history[symbol].append((market_event.datetime, symbol_value))

    @property
    def equity(self):
        equity = self.cash
        for symbol, position in self.positions.items():
            price = self.current_prices.get(symbol, 0)
            equity += position.quantity * price
        return equity

    @property
    def equity_df(self):
        return pd.DataFrame(self.history, columns=["datetime", "equity"]).set_index("datetime")

    @property
    def symbol_equity_df(self) -> Dict[str, pd.DataFrame]:
        retuslts = []
        for symbol, history in self.symbol_equity_history.items():
            df = pd.DataFrame(history, columns=["datetime", symbol]).set_index("datetime").dropna()
            df = df[df[symbol] != 0].sort_index()
            retuslts.append(df)

        return pd.concat(retuslts)

    def on_signal(self, signal_event: SignalEvent):
        symbol = signal_event.symbol
        direction = signal_event.signal_type
        quantity = 10  # signal_event.quantity

        price = self.current_prices.get(symbol, 0)

        if direction == "BUY":
            # Handle BUY or SELL signals with LIMIT price logic
            if self.cash < price * quantity:
                message = f"SIGNAL={direction}  fail at  {quantity} because of cash {self.cash} < {price * quantity}"
                logs.record_log(message=message, level=3)
                return

        elif direction == "SELL" and symbol in self.positions:

            if self.positions[symbol].quantity < quantity:
                message = f"SIGNAL={direction} {symbol} fail at quantity={quantity} because of not enough holdings {self.positions}"
                logs.record_log(message=message, level=3)
                return
        # elif direction == 'HOLDING':
        #     message = f"Unknown signal type: {direction} for {symbol} holding {self.positions} at {signal_event.datetime}  "
        #     logs.record_log(message=message, level=3)
        #     return

        self.events.put(OrderEvent(symbol, "MKT", quantity, direction, signal_event.datetime))

    def on_fill(self, event: FillEvent) -> None:
        """
        Process a FillEvent: deduct cash and update position quantities.
        """

        symbol = event.symbol
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)

        position = self.positions[symbol]
        logs.record_log(position)
        quantity = event.quantity
        price = event.price
        cost = price * quantity
        direction = event.direction
        realized_pnl = 0

        if direction == "BUY":
            commission = self.calculate_buy_commission(cost) if self.commission_rate else 0

            self.cash -= cost + commission
            event.commission = commission

            logs.record_log(f"buy commission: {commission}  cost: {cost}  direction: {direction} cash: {self.cash}")

        elif direction == "SELL":
            commission = self.calculate_sell_commission(cost) if self.commission_rate else 0
            logs.record_log(f"commission={commission}")

            # # Proportional buy commission
            # if position.total_buy_quantity < 0:
            #     logs.record_log(f"not enough quantity to sell {event}")
            #     return

            if position.total_buy_quantity == 0:
                logs.record_log(f'error selling {position} event={event}')
                return
            buy_fee_applied = (quantity / position.total_buy_quantity) * position.total_buy_commission
            gross_pnl = (price - position.avg_price) * quantity
            realized_pnl = gross_pnl - buy_fee_applied - commission

            self.realized_pnl[symbol] = self.realized_pnl.get(symbol, 0.0) + realized_pnl
            self.cash += cost - commission
            event.commission = commission
        else:
            return

        position.update_on_fill(event)

        self.transactions.append(TradeRecord(
            symbol=symbol,
            date=event.datetime,
            direction=direction,
            quantity=quantity,
            price=price,
            commission=commission,
            stamp_tax=self.stamp_duty_rate if direction == "SELL" else 0.0,
            realized_pnl=realized_pnl
        ))

        logs.record_log(
            f"[FILL] {symbol} | Action: {direction} | Qty: {quantity} | Px: {price:.2f} | Cash: {self.cash:.2f}",
            level=1
        )

        # self.trades.append({
        #     'symbol': event.symbol,
        #     'date': event.date,
        #     'price': event.price,
        #     'quantity': event.quantity
        # })

    def calculate_quantity(self, price):
        risk_amount = self.cash * self.risk_pct
        quantity = int(risk_amount // price)
        return max(quantity, 1)

    def calculate_buy_commission(self, amount):
        commission = max(amount * self.commission_rate, self.min_commission)
        return commission

    def calculate_sell_commission(self, amount):
        # logs.record_log(f"amount={amount}, commission_rate={self.commission_rate}, commission={amount*self.commission_rate}")
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp_duty = amount * self.stamp_duty_rate
        transfer_fee = amount * self.transfer_fee_rate
        total_fee = commission + stamp_duty + transfer_fee

        # logs.record_log(f'commission={commission}, stamp_duty={stamp_duty}, transfer_fee={transfer_fee}')
        return total_fee

    def record_daily_snapshot(self, date: datetime) -> None:
        holdings_value = {
            symbol: pos.quantity * self.current_prices.get(symbol, 0.0)
            for symbol, pos in self.positions.items()
        }
        total_equity = sum(holdings_value.values()) + self.cash

        self.history.append((date, total_equity))  # Only once per day
        self.daily_snapshots.append(DailySnapshot(
            date=date,
            cash=self.cash,
            equity=total_equity,
            holdings=holdings_value,
            total_value=total_equity
        ))

        for symbol, value in holdings_value.items():
            if symbol not in self.symbol_equity_history:
                self.symbol_equity_history[symbol] = []
            if not value:
                continue
            self.symbol_equity_history[symbol].append((date, value))
