import argparse
import os
import logging
import sys
import time

import cloudpickle as cp
from cloudburst.client.client import CloudburstConnection
from cloudburst.server.benchmarks.utils import print_latency_stats
import numpy as np

from flow.benchmarks.utils import run_distributed_benchmark
from flow.operators.flow import Flow, FlowType
from flow.optimize import optimize
from flow.types.basic import NumpyType, StrType
from flow.types.table import Row, Table

NUM_USERS = 100000
NUM_PRODUCT_SETS = 1000

USER_SIZE = 512
PRODUCT_SIZE = (2500, 512)

optimize_rules = {
    'fusion': False,
    'compete': False,
    'colocate': False,
    'breakpoint': True,
    'whole': False
}


def pick_category(self, row: Row) -> (str, np.ndarray, str):
    import numpy as np

    category = np.sum(row['recent']) % NUM_PRODUCT_SETS
    category = 'category-' + str(category)
    return row['user'], row[row['user']], category

def get_topk(self, row: Row) -> (int, int, int, int, int):
    users = row['weights']
    products = row[row['category']]

    scores = products @ users
    scores = scores.argsort()[-5:][::-1]

    return scores[0].item(), scores[1].item(), scores[2].item(), \
        scores[3].item(), scores[4].item()

def run(flow, cloudburst, requests, local, sckt=None):
    latencies = []

    if not local:
        print = logging.info

    bench_start = time.time()
    for i in range(requests):
        if i % 100 == 0:
            logging.info(f'On request {i}...')

        inp = Table([('user', StrType), ('recent', NumpyType)])

        uid = np.random.randint(NUM_USERS)
        recent = np.random.randint(0, NUM_PRODUCT_SETS, 5)

        inp.insert([str(uid), recent])

        start = time.time()
        flow.run(inp).get()
        end = time.time()

        latencies.append(end - start)

    bench_end = time.time()

    print_latency_stats(latencies, "E2E", not local, bench_end - bench_start)

    if sckt:
        bts = cp.dumps(latencies)
        sckt.send(bts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs the cascade pipeline.')

    parser.add_argument('-c', '--cloudburst', nargs=1, type=str, metavar='C',
                        help='The address of the Cloudburst ELB',
                        dest='cloudburst', required=True)
    parser.add_argument('-i', '--ip', nargs=1, type=str, metavar='I',
                        help='This machine\'s IP address', dest='ip',
                        required=True)

    parser.add_argument('-r', '--requests', nargs=1, type=int, metavar='R',
                        help='The number of requests to run (required)',
                        dest='requests', required=True)
    parser.add_argument('-t', '--threads', nargs=1, type=int, metavar='T',
                        help='The number of threads to use (required)',
                        dest='threads', required=True)
    parser.add_argument('-l', '--local', nargs=1, type=str, metavar='L',
                        help='Whether to run in local mode (required)',
                        dest='local', required=True)

    args = parser.parse_args()
    print('Connecting to Cloudburst...')
    cloudburst = CloudburstConnection(args.cloudburst[0], args.ip[0])

    flow = Flow('recsys-flow', FlowType.PUSH, cloudburst)
    flow.lookup('user', dynamic=True) \
        .map(pick_category, names=['user', 'weights', 'category']) \
        .lookup('category', dynamic=True) \
        .map(get_topk, names=['1', '2', '3', '4', '5'])

    flow = optimize(flow, rules=optimize_rules)

    print('Creating data...')
    # for i in range(NUM_USERS):
    #     if i % 10000 == 0:
    #         print(f'On user {i}...')

    #     user_vector = np.random.randn(512)
    #     cloudburst.put_object(str(i), user_vector)

    # print('Finished users, starting products...')
    # for i in range(NUM_PRODUCT_SETS):
    #     if i % 100 == 0:
    #         print(f'On product set {i}...')

    #     product_set = np.random.randn(2500, 512)
    #     key = 'category-' + str(i)
    #     cloudburst.put_object(key, product_set)

    print('Deploying flow...')
    flow.deploy()

    print('Starting warmup phase...')
    for i in range(NUM_PRODUCT_SETS):
        if i % 100 == 0:
            print(f'On warmup {i}...')
        uid = np.random.randint(NUM_USERS)
        recent = np.array([i, 0, 0, 0, 0])

        inp = Table([('user', StrType), ('recent', NumpyType)])
        inp.insert([str(uid), recent])

        flow.run(inp).get()

    print('Starting benchmark...')

    local = args.local[0].lower() == 'true'
    if local:
        run(flow, cloudburst, args.requests[0], local)
    else:
        flow.cloudburst = None # Hack to serialize and send flow.
        queue = [flow]
        while len(queue) > 0:
            op = queue.pop(0)
            op.cb_fn = None

            queue.extend(op.downstreams)

        sockets = []

        benchmark_ips = []
        with open('benchmarks.txt', 'r') as f:
            benchmark_ips = [line.strip() for line in f.readlines()]

        sample_input = Table([('user', StrType), ('recent', NumpyType)])
        sample_input.insert([str(1), np.array([1, 2, 3, 4, 5])])

        run_distributed_benchmark(flow, args.requests[0], 'recommender',
                                  args.threads[0], benchmark_ips, sample_input)
