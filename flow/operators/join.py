from typing import Callable

from flow.operators.base import Operator
from flow.types.table import (
    demux_tables,
    deserialize,
    merge_schema,
    merge_tables,
    merge_row,
    Row,
    serialize,
    Table
)

class JoinOperator(Operator):
    def __init__(self,
                 on: str,
                 how: str,
                 flowname: str,
                 init: Callable,
                 sink: list):
        self.__name__ = 'JoinOperator'
        def x(x: int): x + 1 # XXX: This is a hack for registration.
        self.fn = x
        self._setup(flowname)

        self.on = on
        self.how = how

        if self.how == 'right':
            raise FlowError('Right outer joins currently not supported. Please'
                            + ' use left outer joins and switch the join order.')

        self.sink = sink

        class JoinLogic:
            def __init__(self, cloudburst):
                # We pass in None in local mode because we expect the
                # Cloudburst user library in the other case.
                self.preprocess(cloudburst)

            def preprocess(self, _):
                pass

            def run(self, _, on, how, left, right):
                serialized = False
                if type(left) == bytes:
                    left = deserialize(left)
                    right = deserialize(right)
                    serialized = True

                # Note: We currently don't support batching with custom
                # seriralization for joins. Shouldn't be hard to implement but
                # skipping it for expediency.
                batching = False
                if type(left) == list:
                    batching = True
                    _, left = merge_tables(left)
                    mappings, right = merge_tables(right)

                new_schema = merge_schema(left.schema, right.schema)
                result = Table(new_schema)
                ljoin = (how == 'left')
                ojoin = (how == 'outer')

                # Track whether each right row has been inserted for outer
                # joins.
                rindex_map = {}

                for lrow in left.get():
                    lrow_inserted = False

                    idx = 0
                    for rrow in right.get():
                        if lrow[on] == rrow[on]:
                            new_row = merge_row(lrow, rrow, new_schema)
                            result.insert(new_row)
                            lrow_inserted = True

                            rindex_map[idx] = True
                            idx += 1

                    if not lrow_inserted and (ljoin or ojoin):
                        rvals = [None] * len(right.schema)
                        rrow = Row(right.schema, rvals, lrow[Row.qid_key])
                        new_row = merge_row(lrow, rrow, new_schema)
                        result.insert(new_row)

                if ojoin:
                    idx = 0
                    for row in right.get():
                        if idx not in rindex_map:
                            lvals = [None] * len(left.schema)
                            lrow = Row(left.schema, lvals, row[Row.qid_key])
                            new_row = merge_row(lrow, row, new_schema)
                            result.insert(new_row)

                        idx += 1

                if serialized:
                    result = serialize(result)

                if batching:
                    result = demux_tables(result, mappings)

                return result

        self.logic = JoinLogic
        if init is not None:
            self.logic.preprocess = init
        self.exec_args = (self.on, self.how)
