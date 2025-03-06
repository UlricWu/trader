#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : tools.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/3/6 17:13

#!/usr/bin/env python
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
class ModelConfig:

    @classmethod
    @logs.catch()
    def load_file(cls, path, part="train", file_type='yaml'):
        if file_type == 'yaml':
            conf = cls.load_yaml(path)
        else:
            conf = cls.load_file(path)

        if conf is None:
            logs.record_log(f'No config file part={part}found in path={path}', 3)
            return

        model_config = cls()

        for (key, value) in conf.items():
            model_config.__dict__[key] = value

        logs.record_log(model_config.parameters)
        return model_config

    @classmethod
    @logs.catch()
    def load_json(cls, path, part="train"):
        with open(path, 'r', encoding='utf8') as f:
            configs = json.loads(f)
        return configs.get(part)

    @classmethod
    @logs.catch()
    def load_yaml(cls, yaml_file, part="train"):
        with open(yaml_file, 'r', encoding='utf8') as f:
            configs = yaml.safe_load(f)

        return configs.get(part)

    def _set_parameters(self):
        pass

    @property
    def parameters(self):
        return self.__dict__
