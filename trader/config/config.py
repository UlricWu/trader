from dataclasses import dataclass

from enum import Enum


from dataclasses import dataclass
from typing import List


# === Sub-configurations ===

@dataclass
class SystemSettings:
    API_KEY: str = 'test'
    API_SECRET: str = 'test'
    MODE: str = 'train'  # 'train', 'backtest', 'live'


@dataclass
class DataSettings:
    PRICE_ADJUSTMENT: str = 'qfq'  # 'qfq', 'hfq', or None


@dataclass
class ModelSettings:
    SAVE_PATH: str = 'models/'
    FILENAME: str = 'model.pkl'
    AUTO_SAVE: bool = True
    AUTO_LOAD: bool = True


@dataclass
class TradingSettings:
    INITIAL_CASH: float = 100000.0
    COMMISSION_RATE: float = 0.001  # 0.1%
    SLIPPAGE: float = 0.0005  # 5 bps
    LIMIT_UP_DOWN_BUFFER: float = 0.01  # 1% buffer
    SYMBOL_LIST: List[str] = ('000001.SZ', '600519.SH', '300750.SZ')


@dataclass
class CalendarSettings:
    USE_TRADING_CALENDAR: bool = True
    PROVIDER: str = 'tushare'
    TUSHARE_TOKEN: str = 'your-tushare-token'


@dataclass
class DatabaseSettings:
    DB_URI: str = '127.0.0.1:6001'


@dataclass
class LoggingSettings:
    DEBUG: bool = True


# === Master Configuration ===

@dataclass
class CommonConfig:
    system: SystemSettings = SystemSettings()
    data: DataSettings = DataSettings()
    model: ModelSettings = ModelSettings()
    trading: TradingSettings = TradingSettings()
    calendar: CalendarSettings = CalendarSettings()
    database: DatabaseSettings = DatabaseSettings()
    logging: LoggingSettings = LoggingSettings()


# === Different Environments ===

@dataclass
class Local(CommonConfig):
    database: DatabaseSettings = DatabaseSettings(DB_URI='127.0.0.1:6001')
    logging: LoggingSettings = LoggingSettings(DEBUG=True)


@dataclass
class Production(CommonConfig):
    database: DatabaseSettings = DatabaseSettings(DB_URI='remote/db/uri')
    logging: LoggingSettings = LoggingSettings(DEBUG=False)


@dataclass
class Staging(Production):
    logging: LoggingSettings = LoggingSettings(DEBUG=True)