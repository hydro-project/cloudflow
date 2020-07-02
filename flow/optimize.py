from typing import List
import uuid

from flow.operators.agg import AggOperator
from flow.operators.base import Operator
from flow.operators.flow import Flow
from flow.operators.filter import FilterOperator
from flow.operators.groupby import CombineOperator, GroupbyOperator
from flow.operators.join import JoinOperator
from flow.operators.lookup import LookupOperator
from flow.operators.map import MapOperator
from flow.operators.multi import MultiOperator
from flow.types.error import FlowError

DEFAULT_RULES = {
    'fusion': True,
    'compete': False,
    'compete_replicas': 1,
    'colocate': False,
    'breakpoint': True,
    'whole': False
}

def optimize(flow, rules: dict=DEFAULT_RULES):
    for key in DEFAULT_RULES:
        if key not in rules:
            rules[key] = False

    if rules['colocate'] and rules['breakpoint']:
        raise FlowError('Cannot enable the colocate and breakpoint rules'
                        + ' together.')

    optimized = Flow(flow.flowname, flow.typ, flow.cloudburst, flow.source)

    if rules['whole']:
        cloned = optimize(flow, {
            'fusion': False,
            'compete': False,
            'compete_replicas': 1,
            'colocate': False,
            'breakpoint': False,
            'whole': False
        })

        cloned.cloudburst = None # Remove sockets to serialize and send flow.
        queue = [cloned]
        gpu = False
        batching = []
        while len(queue) > 0:
            op = queue.pop(0)
            op.cb_fn = None

            if type(op) != Flow:
                batching.append(op.batching)
            gpu = op.gpu if not gpu else gpu
            queue.extend(op.downstreams)

        if all(batching):
            cloned.batching = True

        optimized.multi([cloned], whole=True)
        multi_op = optimized.downstreams[0]
        multi_op.batching = all(batching)
        multi_op.gpu = gpu

        if gpu:
            multi_op.fn_name += '-gpu'

        return optimized

    ### OPERATOR FUSION ###
    queue = []
    join_tracker = {}
    processed = set()

    for ds in flow.downstreams:
        queue.append((ds, optimized))

    # NOTE: We clone the whole flow regardless. If fusion is turned on,
    # then we will fuse operators, and otherwise, we simply find chains,
    # throw them away, and add operators to the optimized flow.
    while len(queue) > 0:
        op, upstream = queue.pop(0)

        if op.fn_name in processed:
            continue

        chain = find_chain(op)

        if len(chain) == 0 or not rules['fusion']:
            downstreams = op.downstreams
            processed.add(op.fn_name)

            if type(op) == MapOperator:
                marker = upstream.map(op.fn, op.col, op.names,
                                      op.logic.preprocess, op.high_variance,
                                      op.gpu, op.batching, op.multi)
            if type(op) == FilterOperator:
                marker = upstream.filter(op.fn, op.group, op.logic.preprocess)
            if type(op) == GroupbyOperator:
                marker = upstream.gropuby(op.groupby_key, op.logic.preprocess)
            if type(op) == CombineOperator:
                marker = upstream.combine()
            if type(op) == LookupOperator:
                # Merge lookup operators with their successors.
                downstreams = []
                for ds in op.downstreams:
                    if isinstance(ds, MultiOperator):
                        ops = [op] + ds.ops
                    else:
                        ops = [op, ds]
                    marker = upstream.multi(ops)

                    for next_ds in ds.downstreams:
                        queue.append((next_ds, marker))
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
        else:
            marker = upstream.multi(chain)
            downstreams = chain[-1].downstreams

            for op in chain:
                # Set the multi operator to have various properties.
                if op.high_variance:
                    optimized.operators[marker.position].high_variance = True
                if op.gpu:
                    optimized.operators[marker.position].gpu = True

                    # Hack for autoscaling...
                    optimized.operators[marker.position].fn_name += '-gpu'
                if op.batching:
                    optimized.operators[marker.position].batching = True

            if optimized.operators[marker.position].batching:
                for old in chain:
                    if not old.batching:
                        print('Cannot create a fused operator with'
                              + ' batching enabled if all operators do'
                              + ' not batch.')
                        optimized.operators[marker.position].batching = False

        for ds in downstreams:
            queue.append((ds, marker))

    ### LOCALITY BREAKPOINTS ###
    if rules['breakpoint']:
        queue = [optimized]
        processed = set()

        while len(queue) > 0:
            op = queue.pop(0)

            if op.fn_name in processed:
                continue

            # We only set breakpoints if we are in a linear chain portion of the
            # flow. This will only be true if there is only one operator in the
            # queue at a time. After pop, the length should be 0 until we add this
            # op's downstreams.
            if len(queue) == 0:
                if isinstance(op, LookupOperator):
                    op.breakpoint = True
                if isinstance(op, MultiOperator):
                    for sub in op.ops:
                        if isinstance(sub, LookupOperator):
                            op.breakpoint = True

            processed.add(op.fn_name)
            queue.extend(op.downstreams)

    ### COMPETITIVE EXECUTION ###
    if rules['compete']:
        new_ops = []
        for operator in optimized.operators.values():
            if operator.high_variance:
                for downstream in operator.downstreams:
                    if len(downstream.upstreams) > 1:
                        raise RuntimeError("Cannot have a competitive" +
                                           " execution map feed into an " +
                                           "operator with multiple upstreams.")
                    downstream.multi_exec = True

                for _ in range(rules['compete_replicas']):
                    # Create a new operator that is an exact replica.
                    if isinstance(operator, MapOperator):
                        new_op = MapOperator(operator.fn, operator.fntype,
                                             operator.flowname, operator.col,
                                             operator.names,
                                             operator.logic.preprocess,
                                             operator.high_variance,
                                             operator.gpu, operator.batching,
                                             operator.multi, optimized.sink)

                    if isinstance(operator, MultiOperator):
                        new_op = MultiOperator(operator.ops, operator.flowname,
                                               optimized.sink)

                    # Hook it into the DAG by updating all up/downstreams.
                    new_op.downstreams = list(operator.downstreams)
                    new_op.upstreams = list(operator.upstreams)

                    for op in new_op.downstreams:
                        op.upstreams.append(new_op)

                    for op in new_op.upstreams:
                        op.downstreams.append(new_op)

                    new_ops.append(new_op)
        for new_op in new_ops:
            optimized.operators[str(uuid.uuid4())] = new_op

    if rules['colocate']:
        curr_op = optimized

        while len(curr_op.downstreams) > 0:
            if len(curr_op.downstreams) == 1:
                curr_op = curr_op.downstreams[0]
            else: # We only support one colocation for now.
                if not curr_op.supports_broadcast:
                    raise RuntimeError('Unsupported broadcast attempt.')

                colocates = list(map(lambda op: op.fn_name,
                                     curr_op.downstreams))
                optimized.colocates = colocates

                for op in curr_op.downstreams:
                    if not curr_op.supports_broadcast:
                        raise RuntimeError('Unsupported broadcast attempt.')
                    args = list(op.init_args)
                    args[1] = True # Receive broadcast.
                    op.init_args = tuple(args)

                args = list(curr_op.init_args)
                args[0] = True # Send broadcast.
                curr_op.init_args = tuple(args)
                break

    return optimized

def find_chain(start: Operator):
    result = []

    curr_op = start
    while True:
        if len(curr_op.upstreams) == 1: # If this a JoinOperator, don't merge.
            result.append(curr_op)
        else:
            break

        if len(curr_op.downstreams) == 1:
            curr_op = curr_op.downstreams[0]
        else: # End of the flow.
            break

    if len(result) == 1:
        result = []

    return result
