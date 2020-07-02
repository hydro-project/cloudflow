import argparse
import logging
import os
import random
import sys
import time

from cloudburst.server.benchmarks.utils import print_latency_stats
from cloudburst.client.client import CloudburstConnection
import cloudpickle as cp
import numpy as np
from PIL import Image

from flow.benchmarks.utils import run_distributed_benchmark
from flow.operators.flow import Flow, FlowType
from flow.optimize import optimize
from flow.types.basic import StrType
from flow.types.table import Row, Table

optimize_rules = {
    'fusion': False,
    'compete': True,
    'compete_replicas': 2,
    'colocate': False,
    'breakpoint': False,
}

def true_filter(self, row: Row) -> bool:
    return True

def classify_language_init(self, _):
    import fasttext

    self.model = fasttext.load_model('/fastext-classifier.model')

def classify_language(self, table: Table) -> (str, str):
    inputs = [row['classify'] for row in table.get()]

    predicts = self.model.predict(inputs)[0]
    predicts = [label[0].split('_')[-1] for label in predicts]

    result = []
    idx = 0

    for row in table.get():
        result.append([predicts[idx], row['translate']])
        idx += 1

    return result

def filter_french(self, row: Row) -> bool:
    return row['language'] == 'fr'

def filter_german(self, row: Row) -> bool:
    return row['language'] == 'de'

def english_to_french_init(self, _):
    import torch
    self.model = torch.hub.load('pytorch/fairseq', 'conv.wmt14.en-fr', tokenizer='moses', bpe='subword_nmt')

    self.model.eval()

def english_to_french_init_gpu(self, _):
    import os
    import torch
    import cloudpickle as cp
    torch_home = os.getenv('TORCH_HOME')
    french_weights = os.path.join(torch_home, 'french.weights')
    french_model = os.path.join(torch_home, 'french.model')

    with open(french_model, 'rb') as f:
        self.model = cp.loads(f.read())

    self.model.load_state_dict(torch.load(french_weights))
    self.model = self.model.cuda()

    self.model.eval()

def english_to_french(self, table: Table) -> str:
    if type(table) == Table:
        inputs = [row['translate'] for row in table.get()]
    else:
        inputs = [table]

    if len(inputs) > 0:
        return self.model.translate(inputs)
    else:
        return []

def english_to_french_gpu(self, table: Table) -> str:
    inputs = [row['translate'] for row in table.get()]

    if len(inputs) > 0:
        return self.model.translate(inputs)
    else:
        return []

def english_to_german_init(self, _):
    import torch
    self.model = torch.hub.load('pytorch/fairseq', 'conv.wmt14.en-de',
                                tokenizer='moses', bpe='subword_nmt',
                                max_sentences=10)

    self.model.eval()

def english_to_german_init_gpu(self, _):
    import os
    import torch
    import cloudpickle as cp
    torch_home = os.getenv('TORCH_HOME')
    german_weights = os.path.join(torch_home, 'german.weights')
    german_model = os.path.join(torch_home, 'german.model')

    with open(german_model, 'rb') as f:
        self.model = cp.loads(f.read())

    self.model.load_state_dict(torch.load(german_weights))
    self.model = self.model.cuda()

    self.model.eval()

def english_to_german(self, table: Table) -> str:
    if type(table) == Table:
        inputs = [row['translate'] for row in table.get()]
    else:
        inputs = [table]

    if len(inputs) > 0:
        return self.model.translate(inputs)
    else:
        return []

def english_to_german_gpu(self, table: Table) -> str:
    inputs = [row['translate'] for row in table.get()]

    if len(inputs) > 0:
        return self.model.translate(inputs)
    else:
        return []

def run(flow, cloudburst, requests, local, sckt=None):
    schema = [('classify', StrType), ('translate', StrType)]
    french = [
        'Je m\'appelle Pierre.',
        'Comment allez-vous aujourd\'hui?',
        'La nuit est longue et froide, et je veux rentrer chez moi.',
        'Tu es venue a minuit, mais je me suis déja couché.',
        'On veut aller dehors mais il faut rester dedans.'
    ]

    german = [
        'Ich bin in Berliner.',
        'Die katz ist saß auf dem Stuhl.',
        'Sie schwimmt im Regen.',
        'Ich gehe in den Supermarkt, aber mir ist kalt.',
        'Ich habe nie gedacht, dass du Amerikanerin bist.'
    ]

    english = [
        'What is the weather like today?',
        'Why does it rain so much in April?',
        'I like running but my ankles hurt.',
        'I should go home to eat dinner before it gets too late.',
        'I would like to hang out with my friends, but I have to work.'
    ]

    inputs = []
    for _ in range(20):
        table = Table(schema)

        if random.random() < 0.5:
            other = random.choice(french)
        else:
            other = random.choice(german)

        vals = [other, random.choice(english)]
        table.insert(vals)

        inputs.append(table)

    logging.info('Starting benchmark...')

    latencies = []
    bench_start = time.time()
    for i in range(requests):
        if i % 100 == 0:
            logging.info(f'On request {i}...')

        inp = random.choice(inputs)

        start = time.time()
        result = flow.run(inp).get()
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
    parser.add_argument('-g', '--gpu', nargs=1, type=str, metavar='G',
                        help='Whether to run a GPU or CPU benchmark'
                        + ' (required)', dest='gpu', required=True)

    args = parser.parse_args()
    print('Connecting to Cloudburst...')
    cloudburst = CloudburstConnection(args.cloudburst[0], args.ip[0])

    gpu = args.gpu[0].lower() =='true'
    if gpu:
        german_init = english_to_german_init_gpu
        french_init = english_to_french_init_gpu
        german = english_to_german_gpu
        french = english_to_french_gpu
    else:
        german_init = english_to_german_init
        french_init = english_to_french_init
        german = english_to_german
        french = english_to_french

    with open('imagenet_classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    cloudburst.put_object('imagenet-classes', classes)

    flow = Flow('nmt-flow', FlowType.PUSH, cloudburst)
    classified = flow.map(classify_language,
                          init=classify_language_init,
                          names=['language', 'translate'],
                          batching=True)

    french = classified.filter(filter_french) \
        .map(french,
             init=french_init,
             names=['french'],
             gpu=gpu,
             high_variance=True,
             batching=gpu) \
        .filter(true_filter)

    german = classified.filter(filter_german) \
        .map(german,
             init=german_init,
             names=['german'],
             gpu=gpu,
             high_variance=True,
             batching=gpu) \
        .filter(true_filter)

    french.join(german, how='outer')

    print('Optimizing flow...')
    flow = optimize(flow, rules=optimize_rules)

    print('Deploying flow...')
    flow.deploy()

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

        sample_input = Table([('classify', StrType), ('translate', StrType)])
        sample_input.insert(['Je m\'appelle Pierre.', 'How are you?'])

        run_distributed_benchmark(flow, args.requests[0], 'nmt',
                                  args.threads[0], benchmark_ips, sample_input)
