import argparse
import os
import sys
import time

from cloudburst.client.client import CloudburstConnection
from cloudburst.server.benchmarks.utils import print_latency_stats
import numpy as np

from flow.operators.flow import Flow, FlowType
from flow.optimize import optimize
from flow.types.basic import BoolType, BtsType, FloatType, IntType, StrType
from flow.types.table import Row, Table

DATA_SIZES = {
    'small': 10 * 1024,
    'medium': 500 * 1024,
    'large': 10 * 1024 ** 2
}

optimize_rules = {
    'fusion': True,
    'compete': False,
    'colocate': False
}

def run(cloudburst: CloudburstConnection,
        num_requests: int,
        num_fns: int,
        data_size: str,
        do_optimize: bool):

    def fusion_op(self, row: Row) -> bytes:
        return row['data']

    print(f'Creating flow with {num_fns} operators and {data_size}'
          + f' ({DATA_SIZES[data_size]}) inputs.')

    flow = Flow('fusion-benchmark', FlowType.PUSH, cloudburst)

    marker = flow
    for _ in range(num_fns):
        marker = marker.map(fusion_op, names=['data'])

    if do_optimize:
        flow = optimize(flow, rules=optimize_rules)
        print('Flow has been optimized...')

    flow.deploy()
    print('Flow successfully deployed!')

    latencies = []
    inp = Table([('data', BtsType)])
    inp.insert([os.urandom(DATA_SIZES[data_size])])

    print('Starting benchmark...')
    for i in range(num_requests):
        if i % 100 == 0 and i > 0:
            print(f'On request {i}...')

        start = time.time()
        res = flow.run(inp).get()
        end = time.time()

        latencies.append(end - start)

    print_latency_stats(latencies, 'E2E')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Runs the operator fusion
                                     benchmark which uses small, medium, or
                                     large data with a fused or unfused chain
                                     of operators to demonstrate the benefits
                                     of avoiding data shipping between function
                                     stages.''')

    parser.add_argument('-c', '--cloudburst', nargs=1, type=str, metavar='C',
                        help='The address of the Cloudburst ELB',
                        dest='cloudburst', required=True)
    parser.add_argument('-i', '--ip', nargs=1, type=str, metavar='I',
                        help='This machine\'s IP address', dest='ip',
                        required=True)

    parser.add_argument('-r', '--requests', nargs=1, type=int, metavar='R',
                        help='The number of requests to run (required)',
                        dest='requests', required=True)
    parser.add_argument('-l', '--length', nargs=1, type=int, metavar='L',
                        help='The length of the function chain (required)',
                        dest='length', required=True)
    parser.add_argument('-s', '--size', nargs=1, type=str, metavar='S',
                        help='The pre-determined data size to use (required)',
                        dest='size', required=True)
    parser.add_argument('-o', '--optimize', nargs=1, type=str, metavar='O',
                        help='Whether or not to optimize the flow',
                        dest='optimize', required=True)

    args = parser.parse_args()
    if args.size[0] not in DATA_SIZES:
        print(f'Invalid data size: {args.size[0]}')
        print(f'Valid data sizes are {list(DATA_SIZES.keys())}')

        sys.exit(1)

    cloudburst = CloudburstConnection(args.cloudburst[0], args.ip[0])
    print('Successfully connected to Cloudburst')

    run(cloudburst, args.requests[0], args.length[0], args.size[0],
        args.optimize[0].lower() == 'true')
