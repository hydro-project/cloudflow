import os
import shutil

import cloudpickle as cp
import zmq

PORT = 4700

def run_distributed_benchmark(flow, requests, name, threads, benchmark_ips,
                              sample_input):
    ctx = zmq.Context()
    sockets = []

    print('Starting test runs on all threads...')
    for i in range(threads):
        port = (i % 4) + PORT
        ip = benchmark_ips[int(i / 4)]

        address = 'tcp://%s:%d' % (ip, port)
        sckt = ctx.socket(zmq.REQ)
        sckt.connect(address)

        # Send the flow and confirm it is runnable.
        bts = cp.dumps((flow, sample_input))
        sckt.send(bts)

        resp = sckt.recv_string()
        if resp != 'OK':
            raise RuntimeError(resp)

        sockets.append(sckt)

    print('Test runs finished.\nStarting benchmark...')
    inp = cp.dumps((name, requests))
    for sckt in sockets:
        sckt.send(inp)

    # Clean and recreate the data directory.
    if os.path.isdir('data'):
        shutil.rmtree('data')
    os.mkdir('data')

    for idx, sckt in enumerate(sockets):
        latencies_data = sckt.recv()
        with open('data/' + str(idx) + '.data', 'wb') as f:
            f.write(latencies_data)

        sckt.send_string('')
        data = sckt.recv_string()
        with open('data/' + str(idx) + '.txt', 'w') as f:
            f.write(data)

        print(f'Received {idx}...')
