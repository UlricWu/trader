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
    database: str = "db/tutorial.db"
    end_day = "20250205"
    start_day = '20240601'
    ts_code = ["000001.SZ", "000002.SZ", "000003.SZ"]
    symbol_list: List[str] = field(default_factory=lambda: ["MOCK"])
    price_adjustment: str = "qfq"  # qfq or hfq


@dataclass
class TradingSettings:
    INITIAL_CASH: float = 100000.0

    SLIPPAGE: float = 0.0005  # 5 bps
    LIMIT_UP_DOWN_BUFFER: float = 0.01  # 1% buffer
    limit_up_pct: float = 0.10
    limit_down_pct: float = 0.10

    SYMBOL_LIST: List[str] = ('000001.SZ', '600519.SH', '300750.SZ')
    WINDOWS: int = 3
    RISK_PCT: float = 0.01


@dataclass
class StrategySettings:
    short_window: int = 5
    long_window: int = 10


@dataclass
class MLSettings:
    lookback: int = 30

    # model training
    early_stopping_logloss_threshold: float = 0.1
    model_type: str = "RandomForest"
    model_dir: str = "models"

    #  setting
    auto_save: bool = False
    auto_load: bool = False
    auto_version: bool = True
    save_latest: bool = True


@dataclass
class Settings:
    data: DataSettings = field(default_factory=DataSettings)
    trading: TradingSettings = field(default_factory=TradingSettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    model: MLSettings = field(default_factory=MLSettings)


def load_settings(yaml_file: Optional[str] = None) -> Settings:
    if yaml_file:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        return Settings(
            data=DataSettings(**data.get('data', {})),
            trading=TradingSettings(**data.get('trading', {})),
            strategy=StrategySettings(**data.get('strategy', {})),
            model=MLSettings(**data.get('ml', {})),
        )
    else:
        return Settings()
