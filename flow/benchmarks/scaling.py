import argparse
import os
import shutil
import time

from cloudburst.client.client import CloudburstConnection
import cloudpickle as cp
import zmq

from flow.operators.flow import Flow, FlowType
from flow.types.basic import IntType
from flow.types.table import Row, Table

WAIT_TIME = 40 # Number of seconds between load waves.
PORT = 4700

NUM_THREADS = 4

def stage1(self, row: Row) -> int:
    return row['val'] + 1

def stage2(self, row: Row) -> int:
    import time
    time.sleep(.1) # 100ms
    return row['val'] + 1

def run(flow, requests, local, socket):
    latencies = []
    epoch_latencies = []
    epoch_start = time.time()
    epoch_count = 1

    epoch_data = {}

    bench_start = time.time()
    for _ in range(requests):
        start = time.time()
        res = flow.run(inp).get()
        end = time.time()

        latencies.append(end - start)
        epoch_latencies.append(end - start)

        if time.time() - epoch_start > EPOCH_PERIOD:
            epoch = 'EPOCH %d' % (epoch_count)
            print_latency_stats(epoch_latencies, epoch, True, EPOCH_PERIOD)

            epoch_data[epoch_start] = epoch_latencies

            epoch_count += 1
            epoch_latencies = []
            epoch_start = time.time()

    print_latency_stats(latencies, 'FINAL', True, time.time() - bench_start)

    socket.send(cp.dumps(epoch_data))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Runs the scaling
                                     benchmark which measures base latencies
                                     then spikes load and measures latency and
                                     throughput responsiveness.''')

    parser.add_argument('-c', '--cloudburst', nargs=1, type=str, metavar='C',
                        help='The address of the Cloudburst ELB',
                        dest='cloudburst', required=True)
    parser.add_argument('-i', '--ip', nargs=1, type=str, metavar='I',
                        help='This machine\'s IP address', dest='ip',
                        required=True)

    parser.add_argument('-r', '--requests', nargs=1, type=int, metavar='R',
                        help='The number of requests to run (required)',
                        dest='requests', required=True)
    parser.add_argument('-s', '--start', nargs=1, type=float, metavar='S',
                        help='The percentage of benchmarks to baseline with' +
                        '(required)', dest='start', required=True)
    parser.add_argument('-b', '--benchmarks', nargs=1, type=str, metavar='O',
                        help='The name of the file with the benchmark IPs',
                        dest='benchmarks', required=True)

    args = parser.parse_args()

    benchmark_ips = []
    with open(args.benchmarks[0], 'r') as f:
        benchmark_ips = f.readlines()

    cloudburst = CloudburstConnection(args.cloudburst[0], args.ip[0])
    print('Successfully connected to Cloudburst')

    flow = Flow('scaling-benchmark', FlowType.PUSH, cloudburst)
    flow.map(stage1, names=['val']).map(stage2, names=['val'])

    table = Table([('val', IntType)])

    table.insert([1])

    num_bench = len(benchmark_ips)
    num_start = int(start_percent * num_bench)

    flow.cloudburst = None # Hack to serialize and send flow.
    queue = [flow]
    while len(queue) > 0:
        op = queue.pop(0)
        op.cb_fn = None

        queue.extend(op.downstreams)

    flow = cp.dumps(flow)

    sockets = []
    context = zmq.Context(1)

    print(f'Starting baseline with {num_start} nodes...')
    for i in range(num_start):
        for j in range(NUM_THREADS):
            address = ('tcp://%s:%d' % (benchmark_ips[i].strip(), PORT + j))

            sckt = context.socket(zmq.REQ)
            sckt.connect(address)
            sckt.send(flow)

            resp = sckt.recv_string()
            if resp != 'OK':
                raise RuntimeError(f'Unexpected benchmark response: {resp}')

            sckt.send_string(str(int(2.5 * num_requests)))
            sockets.append(sckt)

    time.sleep(WAIT_TIME)

    print(f'Starting high load with remaining ({num_bench - num_start}) nodes...')
    for i in range(num_start, num_bench):
        for j in range(NUM_THREADS):
            address = ('tcp://%s:%d' % (benchmark_ips[i].strip(), PORT + j))

            sckt = context.socket(zmq.REQ)
            sckt.connect(address)
            sckt.send(flow)

            resp = sckt.recv_string()
            if resp != 'OK':
                raise RuntimeError(f'Unexpected benchmark response: {resp}')

            sckt.send_string(str(num_requests))
            sockets.append(sckt)

    if os.path.isdir('data'):
        shutil.rmtree('data')

    os.mkdir('data')
    for idx, sckt in enumerate(sockets):
        data = sckt.recv_string()
        with open('data/' + str(idx) + '.txt', 'w') as f:
            f.write(data)

        sckt.send_string('')
        latencies_data = sckt.recv()
        with open('data/' + str(idx) + '.data', 'wb') as f:
            f.write(latencies_data)

        print(f'Received {idx}...')
