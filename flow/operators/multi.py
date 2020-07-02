import logging
from typing import List

from flow.operators.base import Operator
from flow.types.table import (
    demux_tables,
    deserialize,
    merge_tables,
    serialize
)

class MultiOperator(Operator):
    def __init__(self,
                 ops: List[Operator],
                 whole: bool,
                 flowname: str,
                 sink):
        self.__name__ = 'MultiOperator'
        def x(x: int): x + 1 # XXX: This is a hack for registration.
        self.fn = x
        self._setup(flowname)

        self.ops = ops
        self.whole = whole

        if whole and len(ops) > 1:
            raise FlowError('Cannot execute a whole DAG if multiple operators'
                            + ' are given.')

        if whole:
            logics = ops
        else:
            logics = list(map(lambda op: op.logic, self.ops))

        init_args = list(map(lambda op: op.init_args, self.ops))
        exec_args = list(map(lambda op: op.exec_args, self.ops))
        self.init_args = (logics, self.whole, init_args, exec_args)

        # Clear the metadata in the suboperators to avoid any lingering cruft.
        for op in ops:
            op.upstreams = []
            op.dowwnstreams = []
            op.cloudburst = None

        class MultiLogic:
            def __init__(self, cloudburst, logics, whole, init_args, exec_args):
                self.logics = []
                self.exec_args = exec_args

                self.whole = whole

                if self.whole:
                    flow = logics[0]
                    flow.run = flow.local
                    self.logics = [flow]
                else:
                    for idx, logic in enumerate(logics):
                        obj = logic(*((cloudburst,) + init_args[idx]))
                        self.logics.append(obj)

            def run(self, cloudburst, final, *inp):
                # inp is a tuple because we might take in multiple things for a
                # lookup situation.
                if len(inp) == 1:
                    inp = inp[0]

                prev = inp # inp should either be a Table or a list of Tables.
                if type(inp) == bytes:
                    print('Received a non-batched serialized input.')

                # If the input is a list of Tables, then batching is enabled.
                batching = all([op.batching for op in ops])
                serialized = False
                if batching:
                    if type(prev[0]) == bytes:
                        serialized = True
                        prev = [deserialize(tbl) for tbl in prev]

                    prev, mappings = merge_tables(prev)

                    # This will all be repeated because of the way Cloudburst's
                    # batching works, so we just pick the first one.
                    final = final[0]
                else:
                    if type(prev) == bytes:
                        serialized = True
                        prev = deserialize(prev)

                # NOTE: We currenetly don't support inputs from
                # LookupHelperOperators with batching enabled.
                if type(inp) == tuple:
                    if type(inp[1]) == bytes:
                        sereialized = True
                        inp = (inp[0], deserialize(inp[1]))

                for i in range(len(self.logics)):
                    logic = self.logics[i]

                    if self.whole:
                        # Populate this once for instantiation.
                        if logic.cloudburst is None:
                            queue = [logic]

                            while len(queue) > 0:
                                op = queue.pop(0)
                                op.cloudburst = cloudburst

                                queue.extend(op.downstreams)

                        # prev will never be a tuple with whole beacuse there
                        # will never be a look. See comment at the top of this
                        # function for why inp might be a tuple.
                        args = self.exec_args[i] + (prev,)
                    else:
                        if type(prev) != tuple:
                            args = (cloudburst,) + self.exec_args[i] + (prev,)
                        else:
                            args = (cloudburst,) + self.exec_args[i] + prev

                    prev = logic.run(*args)

                if self.whole:
                    prev = logic.results()[0]

                if batching:
                    if type(prev) == tuple:
                        prev = demux_tables(prev[0], mappings)
                    else:
                        prev = demux_tables(prev, mappings)

                    if serialized:
                        prev = [serialize(tbl) for tbl in prev]
                else:
                    if serialized and not isinstance(prev, tuple):
                        prev = serialize(prev)

                return prev

        self.logic = MultiLogic


    def get_exec_args(self):
        return (self.final,)
