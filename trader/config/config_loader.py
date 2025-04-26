#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : config_loader.py.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/26 20:59
# config_loader.py


# === Settings Models ===
# config_loader.py
import yaml
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataSettings:
    data_path: str = "mock_data.csv"
    symbol_list: List[str] = field(default_factory=lambda: ["MOCK"])
    price_adjustment: str = "qfq"  # qfq or hfq


@dataclass
class TradingSettings:
    INITIAL_CASH: float = 100000.0
    COMMISSION_RATE: float = 0.001  # 0.1%

    SLIPPAGE: float = 0.0005  # 5 bps
    LIMIT_UP_DOWN_BUFFER: float = 0.01  # 1% buffer
    SYMBOL_LIST: List[str] = ('000001.SZ', '600519.SH', '300750.SZ')
    WINDOWS: int = 3
    RISK_PCT: float = 0.01


@dataclass
class ExecutionSettings:
    commission_rate: float = 0.001
    limit_up_pct: float = 0.10
    limit_down_pct: float = 0.10


@dataclass
class StrategySettings:
    short_window: int = 5
    long_window: int = 10


@dataclass
class MLSettings:
    model_path: str = "mock_model.pkl"
    auto_save: bool = False
    auto_load: bool = False


@dataclass
class Settings:
    data: DataSettings = field(default_factory=DataSettings)
    trading: TradingSettings = field(default_factory=TradingSettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    ml: MLSettings = field(default_factory=MLSettings)


def load_settings(yaml_file: Optional[str] = None) -> Settings:
    if yaml_file:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        return Settings(
            data=DataSettings(**data.get('data', {})),
            trading=TradingSettings(**data.get('trading', {})),
            execution=ExecutionSettings(**data.get('execution', {})),
            strategy=StrategySettings(**data.get('strategy', {})),
            ml=MLSettings(**data.get('ml', {})),
        )
    else:
        return Settings()
