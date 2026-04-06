"""
    A simple portable registry that can register and init/run class and functions
    copied and modified from voldemortX/pytorch-auto-drive
"""

class SimpleRegistry(object):
    def __init__(self):
        self._map = {}

    def register(self):
        # @name.register()
        def decorator(func_or_class):
            name = func_or_class.__name__
            if name in self._map.keys():
                raise ValueError(f"{name} is already registered")
            self._map[name] = func_or_class
            return func_or_class
    
        return decorator

    def get(self, name):
        res = self._map.get(name)
        if res is None:
            raise ValueError(f"{name} not found in registry")
        
        return res

    def from_dict(self, dict_params, **kwargs):
        if dict_params is None:
            return None
        dict_params = dict_params.copy()
        dict_params.update(kwargs)
        name = dict_params.pop("name")
        func_or_class = self.get(name)

        try:
            return func_or_class(**dict_params)
        except Exception as e:
            print(f"Error initializing {name} with params {dict_params}: {e}")
            raise e