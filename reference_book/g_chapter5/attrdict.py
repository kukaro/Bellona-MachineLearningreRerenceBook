class AttrDict(dict):
    def __getattr__(self, key):
        if key not in self:
            raise ArithmeticError
        return self[key]

    def __setattr__(self, key, value):
        if key not in self:
            raise AttributeError
        self[key] = value
