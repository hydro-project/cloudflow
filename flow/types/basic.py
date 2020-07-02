from typing import Callable, List

import numpy as np

from flow.types.flow_pb2 import (
    INT, FLOWSTRING, BOOL, BYTES, FLOAT, FLOWNUMPY
)

class Type:
    __name__ = 'Type'
    typ = None

class IntType(Type):
    __name__ = 'Int'
    typ = int
    proto = INT

class StrType(Type):
    __name__ = 'Str'
    typ = str
    proto = FLOWSTRING

class BoolType(Type):
    __name__ = 'Bool'
    typ = bool
    proto = BOOL

class BtsType(Type):
    __name__ = 'Bytes'
    typ = bytes
    proto = BYTES

class FloatType(Type):
    __name__ = 'Float'
    typ = float
    proto = FLOAT

class NumpyType(Type):
    __name__ = 'Numpy'
    typ = np.ndarray
    proto = FLOWNUMPY

class FunctionType(Type):
    def __init__(self, args: List[Type], ret: Type):
        self.__name__ = 'Function'
        self.typ = Callable

        self.args = args
        self.ret = ret

def get_type(typ: type):
    if typ == int:
        return IntType
    if typ == str:
        return StrType
    if typ == bool:
        return BoolType
    if typ == float:
        return FloatType
    if typ == bytes:
        return BtsType
    if typ == np.ndarray:
        return NumpyType

def get_from_proto(typ):
    if typ == INT:
        return IntType
    if typ == FLOWSTRING:
        return StrType
    if typ == BOOL:
        return BoolType
    if typ == FLOAT:
        return FloatType
    if typ == BYTES:
        return BtsType
    if typ == FLOWNUMPY:
        return NumpyType


BASIC_TYPES = [IntType, StrType, BoolType, BtsType, FloatType, NumpyType]
