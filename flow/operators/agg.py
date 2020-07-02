from typing import Callable, List

from flow.operators.base import Operator
from flow.types.error import FlowError
from flow.types.basic import FloatType, get_type, IntType, StrType
from flow.types.table import serialize, deserialize, GroupbyTable, Row, Table

AGGREGATES = ['count', 'min', 'max', 'sum', 'average']

class AggOperator(Operator):
    def __init__(self,
                 flowname: str,
                 aggregate: str,
                 column: str,
                 sink: list):
        if aggregate not in AGGREGATES:
            raise FlowError(f'Unknown aggregate: {aggregate}')
        if aggregate != 'count' and column is None:
            raise FlowError(f'For non-count aggregates, column must be'
                            + ' specified.')

        self.__name__ = 'AggregateOperator'
        def x(x: int): x + 1 # XXX: This is a hack for registration.
        self.fn = x
        self._setup(flowname)

        self.aggregate = aggregate
        self.column = column

        self.sink = sink

        class AggregateLogic:
            def __init__(self, cloudburst):
                self.preprocess(cloudburst)

            def preprocess(self, _):
                pass

            def run(self, cloudburst, aggregate, column, inp):
                serialized = False
                if type(inp) == bytes:
                    serialized = True
                    inp = deserialize(inp)

                if aggregate == 'count':
                    aggfn = self.count
                if aggregate == 'min':
                    aggfn = self.min
                if aggregate == 'max':
                    aggfn = self.max
                if aggregate == 'sum':
                    aggfn = self.sum
                if aggregate == 'average':
                    aggfn = self.average

                if isinstance(inp, GroupbyTable):
                    gb_col = inp.col
                    val, _ = next(inp.get())
                    gb_typ = get_type(type(val))

                    result = Table([(gb_col, gb_typ), (aggregate, FloatType)])

                    for val, tbl in inp.get():
                        agg = aggfn(tbl, column)
                        result.insert([val, float(agg)])
                else:
                    result = Table([(aggregate, FloatType)])
                    result.insert([float(aggnf(inp, column))])

                if serialized:
                    result = serialize(result)

                return result


            def count(self, table, column):
                return table.size()

            def min(self, table, column):
                coltp = type(next(table.get())[column])
                if coltp != int and coltp != float:
                    raise FlowError('Cannot apply aggregate to non-numerical'
                                    + ' field.')

                mn = None
                for row in table.get():
                    if mn is None:
                        mn = row[column]

                    if row[column] < mn:
                        mn = row[column]

                return mn

            def max(self, table, column):
                coltp = type(next(table.get())[column])
                if coltp != int and coltp != float:
                    raise FlowError('Cannot apply aggregate to non-numerical'
                                    + 'field.')

                mx = None
                for row in table.get():
                    if mx is None:
                        mx = row[column]

                    if row[column] > mx:
                        mx = row[column]

                return mx

            def sum(self, table, column):
                coltp = type(next(table.get())[column])
                if coltp != int and coltp != float:
                    raise FlowError('Cannot apply aggregate to non-numerical'
                                    + ' field.')

                sm = 0.0
                for row in table.get():
                    sm += row[column]

                return sm

            def average(self, table, column):
                coltp = type(next(table.get())[column])
                if coltp != int and coltp != float:
                    raise FlowError('Cannot apply aggregate to non-numerical'
                                    + ' field.')

                sm = 0.0
                for row in table.get():
                    sm += row[column]

                return sm / table.size()

        self.logic = AggregateLogic
        self.exec_args = (self.aggregate, self.column)
