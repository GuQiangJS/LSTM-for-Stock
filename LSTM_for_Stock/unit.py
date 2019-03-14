from inspect import signature


def _get_param(cls, con_name):
    sig = signature(cls)
    return sig.parameters.get(con_name)


def get_param_default_value(cls, con_name):
    param = _get_param(cls, con_name)
    if param:
        return param.default
    return None
