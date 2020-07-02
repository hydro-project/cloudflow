from typing import Callable, get_type_hints

from flow.types.basic import (
    BoolType,
    FunctionType,
    get_type,
    IntType,
    StrType,
)

def annotate_function(function: Callable) -> FunctionType:
    hints = get_type_hints(function)

    # We ignore one `self` arg.
    if len(hints) != (function.__code__.co_argcount):
        raise RuntimeError("Missing type annotations on function"
                           + f" {function.__name__}")

    ret_raw = hints['return']
    ret_type = ()
    if type(ret_raw) == tuple:
        for ret_tp in ret_raw:
            ret_type += (get_type(ret_tp),)
    else:
        ret_type += (get_type(ret_raw),)
    del hints['return']
    arg_types = []

    for argname in function.__code__.co_varnames[:function.__code__.co_argcount]:
        # XXX: This is a hack. Can we get around forcing users to write self
        # then ignoring it?
        if argname != 'self':
            typ = get_type(hints[argname])
            arg_types.append(typ)

    return FunctionType(arg_types, ret_type)

def validate_function(function: Callable, typ) -> bool:
    if not isinstance(function, Callable):
        return False

    hints = get_type_hints(function)
    typs = typ.__args__

    if len(hints) != len(typs):
        return False

    if hints['return'] != typs[-1]:
        return False

    del hints['return']
    hint_set = set(hints.values())

    for typ in typs[:-1]:
        hint_set.discard(typ)

    return len(hint_set) == 0
