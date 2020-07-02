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

GAMMA_VALS = {
    'low': 1.0,
    'medium': 2.0,
    'high': 4.0
}

optimize_rules = {
    'fusion': False,
    'compete': True,
    'compete_replicas': 0,
    'colocate': False
}

def run(cloudburst: CloudburstConnection,
        num_requests: int,
        gamma: int,
        num_replicas: int):

    def stage1(self, val: int) -> int:
        return val + 1

    def stage2(self, row: Row) -> float:
        import time
        from scipy.stats import gamma

        delay = gamma.rvs(3.0, scale=row['scale']) * 10 / 1000 # convert to ms
        time.sleep(delay)

        return delay

    def stage3(self, row: Row) -> float:
        return row['val']

    print(f'Creating flow with {num_replicas} replicas and'
          + f' gamma={GAMMA_VALS[gamma]}')

    flow = Flow('fusion-benchmark', FlowType.PUSH, cloudburst)
    flow.map(stage1, col='val') \
        .map(stage2, names=['val'], high_variance=True) \
        .map(stage3, names=['val'])

    optimize_rules['compete_replicas'] = num_replicas
    flow = optimize(flow, rules=optimize_rules)
    print('Flow has been optimized...')

    flow.deploy()
    print('Flow successfully deployed!')

    latencies = []
    inp = Table([('val', IntType), ('scale', FloatType)])
    inp.insert([1, GAMMA_VALS[gamma]])

    print('Starting benchmark...')
    for i in range(num_requests):
        if i % 100 == 0 and i > 0:
            print(f'On request {i}...')

        time.sleep(.300) # Sleep to let the queue drain.
        start = time.time()
        res = flow.run(inp).get()
        end = time.time()

        latencies.append(end - start)

    print_latency_stats(latencies, 'E2E')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Runs the competitive
                                     execution benchmark which uses low,
                                     medium, or high variance with a variable
                                     number of parallel operators to
                                     demonstrate the benefits of
                                     competition.''')

    parser.add_argument('-c', '--cloudburst', nargs=1, type=str, metavar='C',
                        help='The address of the Cloudburst ELB',
                        dest='cloudburst', required=True)
    parser.add_argument('-i', '--ip', nargs=1, type=str, metavar='I',
                        help='This machine\'s IP address', dest='ip',
                        required=True)

    parser.add_argument('-r', '--requests', nargs=1, type=int, metavar='R',
                        help='The number of requests to run (required)',
                        dest='requests', required=True)
    parser.add_argument('-g', '--gamma', nargs=1, type=str, metavar='G',
                        help='The scale parameter of the gamma distribution' +
                        '(required)', dest='gamma', required=True)
    parser.add_argument('-p', '--replicas', nargs=1, type=int, metavar='P',
                        help='The number of competitive replicas to use',
                        dest='replicas', required=True)

    args = parser.parse_args()
    if args.gamma[0] not in GAMMA_VALS:
        print(f'Invalid data size: {args.gamma[0]}')
        print(f'Valid data sizes are {list(GAMMA_VALS.keys())}')

        sys.exit(1)

    cloudburst = CloudburstConnection(args.cloudburst[0], args.ip[0])
    print('Successfully connected to Cloudburst')

    run(cloudburst, args.requests[0], args.gamma[0], args.replicas[0])
