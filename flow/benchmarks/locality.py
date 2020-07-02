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
    'small': 1024,
    'medium': 512 * 1024,
    'large': 1024 * 1024
}

optimize_rules = {
    'fusion': False,
    'compete': False,
    'colocate': False,
    'breakpoint': True
}

NUM_DATA_POINTS = 100

def run(cloudburst: CloudburstConnection,
        num_requests: int,
        data_size: str,
        breakpoint: bool,
        do_optimize: bool):

    print('Creating data...')
    size = DATA_SIZES[data_size]
    for i in range(1, NUM_DATA_POINTS+1):
        arr = np.random.rand(size)
        cloudburst.put_object('data-' + str(i), arr)

    def stage1(self, row: Row) -> (int, str):
        idx = int(row['req_num'] / 10) + 1
        key = 'data-%d' % (idx)

        return idx, key

    def stage2(self, row: Row) -> str:
        import numpy as np
        arr = row[row['key']]

        return float(np.sum(arr))

    print(f'Creating flow with {data_size} ({DATA_SIZES[data_size]}) inputs.')

    flow = Flow('locality-benchmark', FlowType.PUSH, cloudburst)
    flow.map(stage1, names=['index', 'key']) \
        .lookup('key', dynamic=True) \
        .map(stage2, names=['sum'])

    optimize_rules['breakpoint'] = breakpoint
    if do_optimize:
        flow = optimize(flow, rules=optimize_rules)
        print('Flow has been optimized...')

    flow.deploy()
    print('Flow successfully deployed!')

    latencies = []
    inp = Table([('req_num', IntType)])

    if breakpoint:
        print('Starting warmup...')
        for i in range(NUM_DATA_POINTS):
            inp = Table([('req_num', IntType)])
            inp.insert([i * 10])

            res = flow.run(inp).get()

        print('Pausing to let cache metadata propagate...')
        time.sleep(15)

    print('Starting benchmark...')
    for i in range(num_requests):
        if i % 100 == 0 and i > 0:
            print(f'On request {i}...')

        inp = Table([('req_num', IntType)])
        inp.insert([i])

        start = time.time()
        res = flow.run(inp).get()
        end = time.time()

        latencies.append(end - start)

    with open('data.bts', 'wb') as f:
        from cloudburst.shared.serializer import Serializer
        ser = Serializer()
        bts = ser.dump(latencies)
        f.write(bts)

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
    parser.add_argument('-s', '--size', nargs=1, type=str, metavar='S',
                        help='The pre-determined data size to use (required)',
                        dest='size', required=True)
    parser.add_argument('-b', '--breakpoint', nargs=1, type=str, metavar='B',
                        help='Whether to enable the breakpoint optimization (required)',
                        dest='breakpoint', required=True)
    parser.add_argument('-o', '--optimize', nargs=1, type=str, metavar='O',
                        help='Whether or not to optimize the flow',
                        dest='optimize', required=True)

    args = parser.parse_args()
    if args.size[0] not in DATA_SIZES:
        print(f'Invalid data size: {args.size[0]}')
        print(f'Valid data sizes are {list(DATA_SIZES.keys())}')

        sys.exit(1)

    cloudburst = CloudburstConnection(args.cloudburst[0], args.ip[0],
                                      local=True)
    print('Successfully connected to Cloudburst')

    run(cloudburst, args.requests[0], args.size[0],
        args.breakpoint[0] == 'true', args.optimize[0].lower() == 'true')
