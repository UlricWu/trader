# from .config import Local, Production, Staging
import os

from .config_loader import load_settings, Settings
# config_space = os.getenv('CONFIG_SPACE', None)
# if config_space:
#     if config_space == 'LOCAL':
#         auto_config = Local
#     elif config_space == 'STAGING':
#         auto_config = Staging
#     elif config_space == 'PRODUCTION':
#         auto_config = Production
#     else:
#         auto_config = None
#         raise EnvironmentError(f'CONFIG_SPACE is unexpected value: {config_space}')
# else:
#     auto_config = Local

# raise EnvironmentError('CONFIG_SPACE environment variable is not set!')
# config_loader.py
# def get_config():
#     env = os.getenv('ENV', 'local').lower()
#
#     if env == 'local':
#         return Local()
#     elif env == 'staging':
#         return Staging()
#     elif env == 'production':
#         return Production()
#     else:
#         raise ValueError(f"Unknown ENV setting: {env}")
