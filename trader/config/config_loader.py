#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : config_loader.py.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/26 20:59
# config_loader.py
from pydantic import BaseModel
from pydantic_yaml import parse_yaml_file_as
from pathlib import Path
from typing import List


# === Settings Models ===

class SystemSettings(BaseModel):
    api_key: str
    api_secret: str
    mode: str  # train, backtest, live


class DataSettings(BaseModel):
    price_adjustment: str  # qfq, hfq, none


class ModelSettings(BaseModel):
    save_path: str
    filename: str
    auto_save: bool
    auto_load: bool


class TradingSettings(BaseModel):
    initial_cash: float
    commission_rate: float
    slippage: float
    limit_up_down_buffer: float
    symbol_list: List[str]


class CalendarSettings(BaseModel):
    use_trading_calendar: bool
    provider: str
    tushare_token: str


class DatabaseSettings(BaseModel):
    db_uri: str


class LoggingSettings(BaseModel):
    debug: bool


# === Master Settings ===

class Settings(BaseModel):
    system: SystemSettings
    data: DataSettings
    model: ModelSettings
    trading: TradingSettings
    calendar: CalendarSettings
    database: DatabaseSettings
    logging: LoggingSettings


def load_settings(config_path: str = "settings.yaml") -> Settings:
    """Load Settings from YAML"""
    return parse_yaml_file_as(Settings, Path(config_path))
