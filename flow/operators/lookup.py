from flow.operators.base import Operator
from flow.types.error import FlowError
from flow.types.table import deserialize, serialize, Table

# If the dynamic variable is set to be true, the lookup_key is treated as a
# column in the previous function's output's table, which (ideally,
# post-breakpoint) is treated as a reference input to this function's argument.
class LookupOperator(Operator):
    def __str__(self):
        return self.fn_name + '-' + self.lookup_key

    def __repr__(self):
        return self.fn_name + '-' + self.lookup_key

    def __init__(self,
                 flowname: str,
                 lookup_key: str,
                 dynamic: bool,
                 local_dummy: object,
                 sink: list):
        self.__name__ = 'LookupOperator'
        def x(x: int): x + 1 # XXX: This is a hack for registration
        self.fn = x
        self._setup(flowname)

        self.dynamic = dynamic
        self.lookup_key = lookup_key
        self.sink = sink

        class LookupLogic:
            def __init__(self, cloudburst):
                # We pass in None in local mode because we expect the
                # Cloudburst user library in the other case.
                self.preprocess(cloudburst)

            def preprocess(self, _):
                pass

            def run(self, cloudburst, lookup_key, dynamic: bool, input_object, inp: Table):
                from flow.types.basic import get_type

                serialized = False
                if type(inp) == bytes:
                    inp = deserialize(inp)
                    serialized = True

                if cloudburst is None or dynamic:
                    obj = input_object
                    lookup_key = next(inp.get())[lookup_key]
                else:
                    obj = cloudburst.get(lookup_key)

                schema = list(inp.schema)
                schema.append((lookup_key, get_type(type(obj))))

                new_table = Table(schema)
                for row in inp.get():
                    vals = [row[key] for key, _ in inp.schema]
                    vals.append(obj)

                    new_table.insert(vals)

                if serialized:
                    new_table = serialize(new_table)
                return new_table

        self.logic = LookupLogic
        self.local_exec_args = (self.lookup_key, self.dynamic, local_dummy)
        if dynamic:
            self.exec_args = (self.lookup_key, self.dynamic)
        else:
            self.exec_args = (self.lookup_key, self.dynamic, None)

# This operator is only used when there is a dynamic lookup. It is added on to
# the end of the previous operator
class LookupHelperOperator(Operator):
    def __init__(self,
                 flowname: str,
                 column: str,
                 sink: list):
        self.__name__ = 'LookupHelperOperator'
        def x(x: int): x + 1 # XXX: This is a hack for registration
        self.fn = x
        self._setup(flowname)

        self.column = column
        self.sink = sink

        class LookupHelperLogic:
            def __init__(self, cloudburst):
                self.preprocess(cloudburst)

            def preprocess(self, _):
                pass

            def run(self, cloudburst, column, inp):
                from cloudburst.shared.reference import CloudburstReference

                # NOTE: We assume this is uniform for now. May want to look
                # into changing this later.
                key_name = next(inp.get())[column]

                return CloudburstReference(key_name, True), inp

        self.logic = LookupHelperLogic
        self.exec_args = (self.column,)
