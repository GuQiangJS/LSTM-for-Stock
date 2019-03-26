from inspect import signature

@staticmethod
def __get_param(cls, con_name):
    sig = signature(cls)
    return sig.parameters.get(con_name)

@staticmethod
def get_param_default_value(cls, con_name):
    param = __get_param(cls, con_name)
    if param:
        return param.default
    return None
