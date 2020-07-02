import uuid

from flow.types.error import FlowError
from flow.types.utils import validate_function

class Operator():
    def __init__(self):
        self.__name__ = 'Operator'
        raise NotImplementedError(f'{self.name} is an abstract class.' +
                                  ' It cannot be instantiated.')

    def _setup(self, flowname):
        self.registered = False
        self.deployed = False
        self.flowname = flowname

        self.fntype = None

        self.downstreams = []
        self.upstreams = []

        self.inputs = []

        self.multi_exec = False

        self.sink = None # A local sink to capture results if this is the last op.

        self.logic = None # The logic that constitutes this operator.
        self.cb_fn = None # The Cloudburst function.
        self.exec_args = () # The fixed args to pass when executing the function.
        self.local_exec_args = () # The fixed args to pass when executing the function locally.

        self.fn_name = self.__name__ + '_' + flowname + '_' + \
            self.fn.__name__ + '_' + str(uuid.uuid4()).split('-')[0]

        self.init_args = tuple()

        self.breakpoint = False
        self.final = False

        self.supports_broadcast = False

        self.gpu = False
        self.batching = False
        self.high_variance = False

        self.instantiated = None
        self.cloudburst = None

    def __str__(self):
        return self.fn_name

    def __repr__(self):
        return self.fn_name

    def _run(self, local, *inp):
        if local and self.local_exec_args:
            args = tuple(self.local_exec_args)
        else:
            args = tuple(self.get_exec_args())

        args = args + inp

        if local:
            # __init__ has a _ argument because Cloudburst passes in the user
            # library. See if there's a way to get around this hack?
            if not self.instantiated:
                init_args = (self.cloudburst,) + self.init_args
                self.instantiated = self.logic(*init_args)

            result = self.instantiated.run(None, *args)
        else:
            result = self.cb_fn(*args).get()

        self.push_downstream(result, local)

    def get_exec_args(self):
        return self.exec_args

    def deploy(self, cloudburst):
        if self.deployed:
            return [], [], {}, [self.fn_name], {}, [], []

        if not self.registered:
            self.register(cloudburst)

        functions, connections, gpus, batching = [], [], [], []
        # NOTE: What if we need something other than None here? Will need an
        # arg at some point, maybe. This is for multi-execution pick-first
        # result back functions.
        if self.multi_exec:
            functions.append((self.fn_name, [None]))
        else:
            functions.append(self.fn_name)

        if self.gpu:
            gpus.append(self.fn_name)
        if self.batching:
            batching.append(self.fn_name)

        arg_map = {}
        arg_map[self.fn_name] = list(self.get_exec_args())

        registered = {}
        for ds in self.downstreams:
            fn_names, ds_conns, fn_args, starts, ds_registered, ds_gpus, \
                ds_batching = \
                ds.deploy(cloudburst)

            for fn_name in starts:
                connections.append((self.fn_name, fn_name))

            functions += fn_names
            connections += ds_conns
            gpus += ds_gpus
            batching += ds_batching
            arg_map.update(fn_args)
            registered.update(ds_registered)

        self.deployed = True

        # We can number these linearly because the optimization algorithm
        # guarantees that breakpoints will only be at linear points in the
        # flow.
        if self.breakpoint:
            uid = str(len(registered))
            name = self.flowname + '-' + uid
            success, error = cloudburst.register_dag(name, functions,
                                                     connections,
                                                     gpu_functions=gpus,
                                                     batching_functions=batching)

            if not success:
                raise FlowError(str(error))

            registered[name] = arg_map
            return [], [], {}, [], registered, [], []
        else:
            return functions, connections, arg_map, [self.fn_name], \
                registered, gpus, batching

    def register(self, cloudburst):
        cb_fn = cloudburst.register((self.logic, self.init_args), self.fn_name)

        if cb_fn is None:
            raise RuntimeError(f"Unexpected error registering {fn_name}")

        self.registered = True
        self.cb_fn = cb_fn

    def type(self):
        return self.fntype

    def add_downstream(self, op):
        assert isinstance(op, Operator)
        self.downstreams.append(op)

    def add_upstream(self, op):
        assert isinstance(op, Operator)
        self.inputs += [None]
        self.upstreams.append(op)

    def push_downstream(self, result, local):
        if len(self.downstreams) == 0:
            self.sink.append(result)
        else:
            for op in self.downstreams:
                op.recv_downstream(self, result, local)

    def recv_downstream(self, upstream, result, local):
        for idx, op in enumerate(self.upstreams):
            if op == upstream:
                self.inputs[idx] = result

        none_count = len(list(filter(lambda v: v is None, self.inputs)))
        if none_count == 0:
            inp = tuple(self.inputs)
            self.inputs = [None] * len(self.upstreams)
            self._run(local, *inp)
