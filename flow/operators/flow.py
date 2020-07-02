from enum import Enum
from typing import Callable, List
import uuid

from cloudburst.shared.reference import CloudburstReference

from flow.operators.agg import AggOperator
from flow.operators.base import Operator
from flow.operators.filter import FilterOperator
from flow.operators.groupby import CombineOperator, GroupbyOperator
from flow.operators.join import JoinOperator
from flow.operators.lookup import LookupOperator, LookupHelperOperator
from flow.operators.map import MapOperator
from flow.operators.multi import MultiOperator
from flow.types.error import FlowError
from flow.types.table import deserialize, serialize, Table
from flow.types.utils import annotate_function

class FlowType(Enum):
    PUSH = 0
    PULL = 1

class FlowMarker:
    def __init__(self, flow, position: str):
        self.flow = flow
        self.position = position

    def map(self, mapfn: Callable, col: str=None, names: List[str]=[], init:
            Callable=None, high_variance: bool=False, gpu: bool=False,
            batching: bool=False, multi: bool=False):
        return self.flow.map(mapfn, col, names, init, high_variance, gpu,
                             batching, multi, self)

    def filter(self, filterfn: Callable, group=False, init: Callable=None):
        return self.flow.filter(filterfn, group, init, self)

    def groupby(self, gb_key: str, init: Callable=None):
        return self.flow.groupby(gb_key, init, self)

    def combine(self):
        return self.flow.combine(self)

    def join(self, other, on: str="qid", how: str="inner", init: Callable=None):
        return self.flow.join(other, on, how, init, self)

    def multi(self, ops, whole: bool=False):
        return self.flow.multi(ops, whole, self)

    def lookup(self, key, dynamic: bool=False, dummy=None):
        return self.flow.lookup(key, dynamic, dummy, self)

    def agg(self, aggfn, col):
        return self.flow.agg(aggfn, col, self)

    def extend(self, other):
        assert isinstance(other, Flow)
        return self.flow.extend(other, self)

class Flow(Operator):
    def __init__(self, name: str, stream_typ: FlowType, cloudburst, source=None):
        if stream_typ == FlowType.PULL:
            raise RuntimeError("PULL streams are not yet implemented.")

        self.__name__ = "Flow"
        self.flowname = name
        def x(x: int): x + 1 # XXX: This is a hack for registration.
        self.fn = x

        self._setup(self.flowname)

        self.cloudburst = cloudburst

        self.typ = stream_typ
        self.source = source

        self.operators = {}

        self.sink = []

        self.registered = {}
        self.deployed = False

        self.colocates = []

        self.cid = None

    def local(self, inp=None):
        if inp is None:
            raise RuntimeError("PULL streams are not supported in local mode.")

        self.push_downstream(inp, True)

    def register(self):
        self.registered = True
        for op in self.operators:
            self.operators[op].register(self.cloudburst)

        return # Flow operators don't need to be registered.

    def exec(self, inp):
        if not self.registered:
            self.register(self.cloudburst)

        self.push_downstream(inp, False)

    def deploy(self):
        queue = [self]

        # Once the DAG is set, mark the final operator as the final. This helps
        # any type conversions from tuples to Table (right now, only used for
        # the MultiOperator).
        num_ends = 0
        seen = set()
        while len(queue) > 0:
            op = queue.pop(0)

            if len(op.downstreams) == 0 and op.fn_name not in seen:
                num_ends += 1
                op.final = True

                seen.add(op.fn_name)

            queue.extend(op.downstreams)

        if num_ends > 1:
            raise FlowError('You must converge all of your operators to a'
                            + ' single output.')

        functions, connections, gpus, batching = [], [], [], []

        for downstream in self.downstreams:
            fns, conns, args, _, registered, ds_gpus, ds_batching = \
                downstream.deploy(self.cloudburst)
            functions += fns
            connections += conns
            gpus += ds_gpus
            batching += ds_batching
            self.registered = registered

        if len(functions) > 0:
            uid = str(len(registered))
            name = self.flowname + '-' + uid
            registered[name] = args

            success, error = self.cloudburst.register_dag(
                name, functions, connections, colocated=self.colocates,
                gpu_functions=gpus, batching_functions=batching)

            if not success:
                raise FlowError(str(error))

        self.deployed = True

    def run(self, inp):
        template = self.flowname + '-%d'

        if not self.cid:
            cid = str(uuid.uuid4()).split('-')[0]
            self.cid = cid

        # if type(inp) == Table:
        #     inp = serialize(inp)
        # elif type(inp) == str:
        #     inp = CloudburstReference(inp, deserialize=False)

        cont = None
        for idx in range(len(self.registered)):
            dag_name = template % (idx)
            args = self.registered[dag_name].copy()

            if idx == len(self.registered) - 1:
                for ds in self.downstreams:
                    fn_args = args[ds.fn_name].copy()
                    fn_args.append(inp)
                    args[ds.fn_name] = fn_args

                cont = self.cloudburst.call_dag(dag_name, args,
                                                continuation=cont,
                                                client_id=self.cid)
            else:
                cont = self.cloudburst.call_dag(dag_name, args,
                                                continuation=cont,
                                                dry_run=True)
        return cont

    def map(self, mapfn: Callable, col: str=None, names: List[str]=[],
            init: Callable=None, high_variance: bool=False, gpu: bool=False,
            batching: bool=False, multi: bool=False, marker: FlowMarker=None):
        fntype = annotate_function(mapfn)
        operator = MapOperator(mapfn, fntype, self.flowname, col, names, init,
                               high_variance, gpu, batching, multi, self.sink)
        marker = [marker] if marker is not None else []

        return self._add_operator(operator, marker)

    def filter(self, filterfn: Callable, group=False, init: Callable=None,
               marker: FlowMarker=None):
        fntype = annotate_function(filterfn)
        operator = FilterOperator(filterfn, fntype, self.flowname, group,
                                  init, self.sink)
        marker = [marker] if marker is not None else []

        return self._add_operator(operator, marker)

    def groupby(self, gb_key: str, init: Callable=None, marker: FlowMarker=None):
        operator = GroupbyOperator(self.flowname,gb_key, init, self.sink)
        marker = [marker] if marker is not None else []

        return self._add_operator(operator, marker)

    def combine(self, marker: FlowMarker=None):
        operator = CombineOperator(self.flowname, self.sink)
        marker = [marker] if marker is not None else []

        return self._add_operator(operator, marker)

    def multi(self, ops: List[Operator], whole: bool=False, marker:
              FlowMarker=None):
        operator = MultiOperator(ops, whole, self.flowname, self.sink)
        marker = [marker] if marker is not None else []

        return self._add_operator(operator, marker)

    def join(self, other: FlowMarker, on: str="qid", how: str="inner",
             init: Callable=None, marker: FlowMarker=None):
        if marker is None:
            raise RuntimeError("We currently do not support joining two input"
                               + " flows.")
        operator = JoinOperator(on, how, self.flowname, init, self.sink)
        marker = [marker, other] if marker is not None else [other]

        return self._add_operator(operator, marker)

    def lookup(self, key: str, dynamic: bool=False, dummy: object=None, marker:
               FlowMarker=None):
        if dynamic:
            helper = LookupHelperOperator(self.flowname, key, self.sink)
            operator = LookupOperator(self.flowname, key, dynamic, dummy,
                                      self.sink)

            # Rewrite previous operator as a multi operator with the helper
            # tucked into it. First, remove the previous operator from all its
            # upstreams.
            if marker:
                prev_op = self.operators[marker.position]
                del self.operators[marker.position]

                ops = [prev_op, helper]
            else:
                ops = [helper]
                prev_op = self

            # Then get a list of all the markers associated with the previous
            # operator's upstreams.
            markers = []
            for us in prev_op.upstreams:
                for key in self.operators:
                    if self.operators[key] == us:
                        new_marker = FlowMarker(self, key)
                        markers.append(new_marker)

            for us in prev_op.upstreams:
                us.downstreams.remove(prev_op)

            # Then create a MultiOperator with the two operators packed
            # together.
            new_prev = MultiOperator(ops, False, self.flowname,
                                     self.sink)

            # Remove the old operator completely and add the new one into the
            # flow.
            marker = self._add_operator(new_prev, markers)
        else:
            operator = LookupOperator(self.flowname, key, dynamic, dummy,
                                      self.sink)

        marker = [marker] if marker is not None else []

        return self._add_operator(operator, marker)

    def agg(self, aggfn: str, col: str, marker: FlowMarker=None):
        operator = AggOperator(self.flowname, aggfn, col, self.sink)

        marker = [marker] if marker is not None else []
        return self._add_operator(operator, marker)

    def extend(self, other, marker: FlowMarker=None):
        assert isinstance(other, Flow)

        # Check the current flow to make sure it has only one final operator.
        queue = [self]
        seen = set()
        num_ends = 0
        sinks = []
        join_tracker = {}

        while len(queue) > 0:
            op = queue.pop(0)
            if op.fn_name not in seen and len(op.downstreams) == 0:
                num_ends += 1

            seen.add(op.fn_name)
            queue.extend(op.downstreams)

        # If there are multiple operators with 0 downstreams, raise an error.
        if num_ends > 1:
            raise FlowError('Cannot extend a flow when there are multiple'
                            ' unmerged operators. Flow can only have one sink.')

        # Iterate over everything in the
        queue = list(other.downstreams)
        upstream_map = {}
        seen = set()

        for op in other.downstreams:
            upstream_map[op.fn_name] = marker if marker else self

        while len(queue) > 0:
            op = queue.pop(0)

            if op.fn_name in seen:
                continue

            seen.add(op.fn_name)
            next_marker = upstream_map[op.fn_name]

            if type(op) == MapOperator:
                marker = next_marker.map(op.fn, op.col, op.names,
                                      op.logic.preprocess, op.high_variance,
                                      op.gpu, op.batching, op.multi)
            if type(op) == FilterOperator:
                marker = next_marker.filter(op.fn, op.group, op.logic.preprocess)
            if type(op) == GroupbyOperator:
                marker = next_marker.gropuby(op.groupby_key, op.logic.preprocess)
            if type(op) == CombineOperator:
                marker = next_marker.combine()
            if type(op) == LookupOperator:
                # Merge lookup operators with their successors.
                marker = op.lookup(op.lookup_key, op.dynamic, op.dummy)
            if type(op) == AggOperator:
                marker = upstream.agg(op.aggregate, op.column)
            if type(op) == MultiOperator:
                # This will only happen in the case where the previous operator
                # was a LookupHelperOperator combined with something else.
                marker = upstream.multi(op.ops)
            if type(op) == JoinOperator:
                if op.fn_name not in join_tracker:
                    join_tracker[op.fn_name] = upstream
                    downstreams = []
                    processed.discard(op.fn_name)
                else:
                    other = join_tracker[op.fn_name]
                    marker = other.join(upstream, op.on, op.how,
                                        op.logic.preprocess)
            for ds in op.downstreams:
                upstream_map[ds.fn_name] = marker

            queue.extend(op.downstreams)


    def results(self):
        results = list(self.sink)
        self.sink.clear() # Empty the result set.
        return results

    def _add_operator(self, operator, markers: List[FlowMarker]):
        opid = str(uuid.uuid4())
        self.operators[opid] = operator

        if len(markers) > 0:
            marker_ops = list(map(lambda marker:
                                  self.operators[marker.position], markers))
        else:
            marker_ops = [self]

        for marker_op in marker_ops:
            operator.add_upstream(marker_op)
            marker_op.add_downstream(operator)

        return FlowMarker(self, opid)

    def __str__(self):
        res = ''
        queue = list(self.downstreams)
        seen = set()

        while len(queue) > 0:
            op = queue.pop(0)
            if op.fn_name not in seen:
                res += op.fn_name + ' ds: ' + \
                    str(list(map(lambda o: o.fn_name, op.downstreams))) + '\n'

                if isinstance(op, MultiOperator):
                    res += ('\t ops:' + str(op.ops) + '\n')

                queue += op.downstreams
                seen.add(op.fn_name)

        return res[:-1] # Removes training newline.

    def __repr__(self):
        return str(self)
