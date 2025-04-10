#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : tools.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/3/6 17:13

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : tools.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/3/6 15:49
import json

import yaml
from functools import wraps

from .logs import logs


def Constant(cls):
    @wraps(cls)
    def new_setattr(self, name, value):
        raise Exception('const : {} can not be changed'.format(name))

    cls.__setattr__ = new_setattr
    return cls


@Constant
class Config(object):
    @classmethod
    def load_config(cls, path, part='train'):
        with open(path, 'r') as f:
            configs = yaml.safe_load(f)

        model_config = cls()

        for (key, value) in configs.get(part, {}).items():
            model_config.__dict__[key] = value

        logs.record_log(model_config.parameters)
        return model_config

    @property
    def parameters(self):
        return self.__dict__
