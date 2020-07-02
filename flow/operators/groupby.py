from typing import Callable

from flow.operators.base import Operator
from flow.types.table import serialize, deserialize, GroupbyTable, Table

class GroupbyOperator(Operator):
    def __init__(self,
                 flowname: str,
                 groupby_key: str,
                 init: Callable,
                 sink: list):
        self.__name__ = 'GroupbyOperator'
        def x(x: int): x + 1 # XXX: This is a hack for registration.
        self.fn = x
        self._setup(flowname)

        self.groupby_key = groupby_key
        self.sink = sink

        class GroupbyLogic:
            def __init__(self, cloudburst):
                # We pass in None in local mode because we expect the
                # Cloudburst user library in the other case.
                self.preprocess(cloudburst)

            def preprocess(self, _):
                pass

            def run(self, _, col: str, inp: Table):
                serialized = False
                if type(inp) == bytes:
                    serialized = True
                    inp = deserialize(inp)

                gb_table = GroupbyTable(inp.schema, col)

                for row in inp.get():
                    gb_table.add_row(row)

                if serialized:
                    gb_table = serialize(gb_table)

                return gb_table

        self.logic = GroupbyLogic
        if init is not None:
            self.logic.preprocess = init
        self.exec_args = (self.groupby_key,)

class CombineOperator(Operator):
    def __init__(self,
                 flowname: str,
                 sink: list):
        self.__name__ = 'CombineOperator'
        def x(x: int): x + 1 # XXX: This is a hack for registration.
        self.fn = x
        self._setup(flowname)

        self.sink = sink

        class CombineLogic:
            def __init__(self, _):
                # We pass in None in local mode because we expect the
                # Cloudburst user library in the other case.
                self.preprocess(None)

            def preprocess(self, _):
                pass

            def run(self, _, inp: GroupbyTable):
                result = Table(inp.schema)

                for group, gtable in inp.get():
                    for row in gtable.get():
                        result.insert(row)
                return result

        self.logic = CombineLogic
