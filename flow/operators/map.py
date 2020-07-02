from typing import Callable, List

from flow.operators.base import Operator
from flow.types.basic import FunctionType
from flow.types.error import FlowError
from flow.types.table import (
    demux_tables,
    deserialize,
    GroupbyTable,
    merge_tables,
    Row,
    serialize,
    Table
)

class MapOperator(Operator):
    def __init__(self,
                 fn: Callable,
                 fntype: FunctionType,
                 flowname: str,
                 col: str,
                 names: List[str],
                 init: Callable,
                 high_variance: bool,
                 gpu: bool,
                 batching: bool,
                 multi: bool,
                 sink: list):
        self.__name__ = 'MapOperator'
        self.fn = fn
        self._setup(flowname)

        self.fntype = fntype
        self.sink = sink
        self.col = col
        self.names = names
        self.high_variance = high_variance
        self.gpu = gpu
        self.batching = batching
        self.multi = multi

        self.supports_broadcast = True

        if col is not None: # Map over column not over whole row.
            if len(names) >= 2: # We can only rename one column here
                raise FlowError('Map over a column cannot rename multiple'
                                + ' columns.')
            if gpu:
                raise FlowError('You cannot do a column map with batching.')
        else:
            if len(names) != 0 and len(names) != len(fntype.ret):
                raise FlowError('Map over row must have same number of columns'
                                + ' as function outputs.')

        class MapLogic:
            def __init__(self, cloudburst, send_broadcast, recv_broadcast,
                         batching, multi):
                # We pass in None in local mode because we expect the
                # Cloudburst user library in the other case.
                self.send_broadcast = send_broadcast
                self.recv_broadcast = recv_broadcast
                self.batching = batching
                self.multi = multi

                self.preprocess(cloudburst)

            def preprocess(self, _):
                pass

            def run(self, cloudburst, fn, fntype, col, names, inp):
                # Merge all of the tables.
                serialized = False
                batching = self.batching and isinstance(inp, list)
                if batching:
                    if type(inp[0]) == bytes:
                        inp = [deserialize(tbl) for tbl in inp]
                        serialized = True

                    # inp will be a list of Tables. If it not, this is part of
                    # a MultiOperator, and everything is taken care of for us.
                    merged, mappings = merge_tables(inp)
                    inp = merged

                    # This will all be repeated because of the way Cloudburst's
                    # batching works, so we just pick the first one. But we
                    # check because even with batching enabled, in a multi
                    # operator, we will not have to deal with this.
                    if type(fn) == list:
                        fn = fn[0]
                    if type(fntype) == list:
                        fntype = fntype[0]
                    if type(col) == list:
                        col = col[0]
                    if type(names) == list and type(names[0]) == list:
                        names = names[0]
                else:
                    if type(inp) == bytes:
                        inp = deserialize(inp)
                        serialized = True

                schema = []
                if col is None:
                    if len(names) != 0:
                        schema = list(zip(names, fntype.ret))
                    else:
                        for i in range(len(fntype.ret)):
                            schema.append((str(i), fntype.ret[i]))
                else:
                    for name, tp in inp.schema:
                        if name != col:
                            schema.append((name, tp))
                        else:
                            if len(names) != 0:
                                schema.append((names[0], fntype.ret[0]))
                            else:
                                schema.append((name, fntype.ret[0]))

                if isinstance(inp, GroupbyTable):
                    result = GroupbyTable(schema, inp.col)
                    for group, gtable in inp.get():
                        result.add_group(group, self.run(fn, fntype, col, gtable))
                else:
                    result = Table(schema)

                    if self.batching or self.multi:
                        res = fn(self, inp)
                        for val in res:
                            if type(val) == tuple:
                                val = list(val)
                            elif type(val) != list:
                                val = [val]

                            result.insert(val)
                    else:
                        for row in inp.get():
                            if col is None:
                                vals = fn(self, row)
                                if type(vals) == tuple:
                                    vals = list(vals)
                                elif type(vals) != list:
                                    vals = [vals]

                                result.insert(vals, row[Row.qid_key])
                            else:
                                val = fn(self, row[col])
                                new_vals = []
                                for name, _ in inp.schema:
                                    if name == col:
                                        new_vals.append(val)
                                    else:
                                        new_vals.append(row[name])

                                result.insert(new_vals, row[Row.qid_key])

                if batching: # Unmerge all the tables.
                    tables = demux_tables(result, mappings)
                    result = tables

                    if serialized:
                        result = [serialize(tbl) for tbl in result]
                else:
                    if serialized:
                        result = serialize(result)

                if self.send_broadcast:
                    import uuid
                    uid = str(uuid.uuid4())
                    cloudburst.put(uid, result)
                    result = uid

                return result

        self.logic = MapLogic
        if init is not None:
            self.logic.preprocess = init
        self.exec_args = (self.fn, self.fntype, self.col, self.names)
        self.init_args = (False, False, self.batching, self.multi)
