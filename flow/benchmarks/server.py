import logging
import os
import sys
import time

from cloudburst.client.client import CloudburstConnection
from cloudburst.server.benchmarks.utils import print_latency_stats
import cloudpickle as cp
import zmq

from flow.benchmarks import scaling
from flow.benchmarks.utils import PORT
from flow.pipelines import cascade, nmt, recommender, video
from flow.types.basic import IntType
from flow.types.table import Table


LOG_FILE = 'log_flow.txt'

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s %(message)s')

EPOCH_PERIOD = 5 # 5 seconds

def main():
    function_addr = os.getenv('FUNCTION_ADDR')
    thread_id = int(os.getenv('THREAD_ID'))
    my_ip = sys.argv[1]

    cloudburst = CloudburstConnection(function_addr, my_ip, thread_id)

    # Set up ZMQ socket.
    address = ('tcp://*:%d' % (PORT + thread_id))
    context = zmq.Context(1)
    socket = context.socket(zmq.REP)
    socket.bind(address)

    # Receive and configure Flow object.
    bts = socket.recv()
    flow, inp = cp.loads(bts)
    flow.cloudburst = cloudburst
    logging.info(f'Received a flow:\n{flow}')

    # Test the flow to make sure it works.
    res = flow.run(inp)
    logging.info(f'Running test query: {res.obj_id}')
    res = res.get()
    logging.info(f'Test query result was: {res}')
    socket.send_string('OK')

    while True:
        pipeline, num_requests = cp.loads(socket.recv())
        logging.info(f'Starting benchmark {pipeline}: {num_requests} requests')

        if pipeline == 'cascade':
            cascade.run(flow, cloudburst, num_requests, False, socket)
        if pipeline == 'nmt':
            nmt.run(flow, cloudburst, num_requests, False, socket)
        if pipeline == 'scaling':
            scaling.run(flow, cloudburst, num_requests, False, socket)
        if pipeline == 'video':
            video.run(flow, cloudburst, num_requests, False, socket)
        if pipeline == 'recommender':
            recommender.run(flow, cloudburst, num_requests, False, socket)

        socket.recv_string()
        with open(LOG_FILE, 'r') as f:
            data = f.read()
            socket.send_string(data)

if __name__ == '__main__':
    main()
