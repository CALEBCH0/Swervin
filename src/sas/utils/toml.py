import tomli
from os import path

def load_toml(fn):
    with open(fn, "rb") as fn:
        return tomli.load(fn)


class Config:
    def __init__(self, toml_data):
        for key, value in toml_data.items():
            if isinstance(value, dict):
                value = Config(value)
            if value == "":
                value = None
            setattr(self, key, value)
        self.join_engine_paths()

    def join_engine_paths(self):
        # Check if 'path_engines' exists
        if hasattr(self, 'path_engines'):
            path_engines = getattr(self, 'path_engines')
            if path_engines:
                for attr in dir(self):
                    current_value = getattr(self, attr)
                    if current_value and isinstance(current_value, str) and current_value.endswith('.engine'):
                        setattr(self, attr, path.join(path_engines, current_value))