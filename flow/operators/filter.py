from typing import Callable

from flow.operators.base import Operator
from flow.types.basic import FunctionType
from flow.types.table import (
    demux_tables,
    deserialize,
    GroupbyTable,
    merge_tables,
    serialize,
    Table
)

class FilterOperator(Operator):
    def __init__(self,
                 fn: Callable,
                 fntype: FunctionType,
                 flowname: str,
                 group: bool,
                 init: Callable,
                 sink: list):
        self.__name__ = 'FilterOperator'
        self.fn = fn
        self._setup(flowname)

        self.fntype = fntype
        self.sink = sink
        self.batching = True

        self.group = group

        class FilterLogic:
            def __init__(self, cloudburst):
                # We pass in None in local mode because we expect the
                # Cloudburst user library in the other case.
                self.preprocess(cloudburst)

            def preprocess(self, _):
                pass

            def run(self, _, fn, group, inp):
                batching = isinstance(inp, list)
                serialized = False

                if batching:
                    if type(inp[0]) == bytes:
                        serialized = True
                        inp = [deserialize(tbl) for tbl in inp]
                else:
                    if type(inp) == bytes:
                        serialized = True
                        inp = deserialize(inp)


                if batching:
                    # Because we have batching enabled by default, we have to
                    # assume these are lists if these are not merged into a multi
                    # operator. We have to check these because a whole flow
                    # operator will not have lists even when batching is
                    # enabled.
                    if type(group) == list:
                        group = group[0]

                    if type(fn) == list:
                        fn = fn[0]
                    inp, mappings = merge_tables(inp)

                if group and not isinstance(inp, GroupbyTable):
                    raise RuntimeError("Can't run a group filter over a non-grouped"
                                       + " table.")

                if group:
                    result = GroupbyTable(inp.schema, inp.col)
                    for group, gtable in inp.get():
                        if fn(self, next(gtable.get())):
                            result.add_group(group, gtable)
                else:
                    result = Table(inp.schema)
                    for row in inp.get():
                        if fn(self, row):
                            result.insert(row)

                if batching:
                    result = demux_tables(result, mappings)
                    if serialized:
                        result = [serialize(tbl) for tbl in result]
                else:
                    if serialized:
                        result = serialize(result)

                return result

        self.logic = FilterLogic
        if init is not None:
            self.logic.preprocess = init
        self.exec_args = (self.fn, self.group)
