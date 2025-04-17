from dataclasses import dataclass


@dataclass
class Common(object):
    API_KEY = 'test'
    API_SECRET = 'test'

    # def __init__(self):
    #     pass

    def get_config(self):
        # return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return {k for k in dir(self) if not k.startswith("_")}

    def from_config(self, configs):
        for k, v in configs.items():
            setattr(self, k.upper(), v)


class Local(Common):
    DB_URI = '127.0.0.1:6001'
    DEBUG = True


class Production(Common):
    DB_URI = 'remote/db/uri'
    DEBUG = False


class Staging(Production):
    DEBUG = True


# schema_config.py
# !config.py

from dataclasses import dataclass


@dataclass
class StrategyConfig:
    window_size: int = 100  # rolling window size for training
    long_threshold: float = 0.01  # signal threshold to go long
    short_threshold: float = -0.01  # signal threshold to go short
    rolling_window: int = 20
