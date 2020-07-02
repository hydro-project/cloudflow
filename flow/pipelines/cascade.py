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
from flow.types.basic import NumpyType
from flow.types.table import Row, Table

optimize_rules = {
    'fusion': False,
    'compete': False,
    'compete_replicas': 0,
    'colocate': False,
    'breakpoint': False,
    'whole': True
}

def transform_init(self, _):
    from torchvision import transforms

    self.transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

def transform(self, row: Row) -> np.ndarray:
    res = self.transform(row['img']).detach().numpy()
    return res

def transform_batch(self, table: Table) -> np.ndarray:
    return [self.transform(row['img']).detach().numpy() for row in table.get()]

def resnet_init(self, _): # args are self and cloudburst user library
    import os

    import torch
    import torchvision

    tpath = os.path.join(os.getenv('TORCH_HOME'), 'checkpoints')
    if 'resnet101.model' in os.listdir(tpath):
        self.resnet = torch.load(os.path.join(tpath, 'resnet101.model'))
    else:
        self.resnet = torchvision.models.resnet101(pretrained=True)

    self.resnet.eval()


def resnet_init_gpu(self, _): # args are self and cloudburst user library
    import os

    import torch
    import torchvision

    tpath = os.path.join(os.getenv('TORCH_HOME'), 'checkpoints')
    if 'resnet101.model' in os.listdir(tpath):
        self.resnet = torch.load(os.path.join(tpath, 'resnet101.model'))
    else:
        self.resnet = torchvision.models.resnet101(pretrained=True)

    self.resnet = self.resnet.cuda()
    self.resnet.eval()


def resnet_model(self, row: Row) -> (np.ndarray, int, float):
    """
    ResNet101 for image classification on ResNet
    """
    import torch

    img_t = torch.from_numpy(row['img'])

    batch_t = torch.unsqueeze(img_t, 0)
    out = self.resnet(batch_t)
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    p_2 = percentage.detach().numpy()

    return row['img'], \
        indices.detach().numpy()[0][0].item(), \
        p_2[indices[0][0]].item()


def resnet_model_gpu(self, table: Table) -> (np.ndarray, int, float):
    """
    ResNet101 for image classification on ResNet
    """
    import torch

    originals = [row['img'] for row in table.get()]
    inputs = [torch.from_numpy(img) for img in originals]
    inputs = torch.stack(inputs, dim=0).cuda()

    out = self.resnet(inputs)
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    p_2 = percentage.cpu().detach().numpy()
    indicies = indices.cpu().detach().numpy()

    result = []
    for i in range(len(originals)):
        index = indices[i][0].item()
        perc = p_2[indices[i][0]].item()
        img = originals[i]

        result.append([img, index, perc])

    return result


def inceptionv3_init(self, _):
    import os

    import torch
    import torchvision

    tpath = os.path.join(os.getenv('TORCH_HOME'), 'checkpoints')
    if 'inception_v3.model' in os.listdir(tpath):
        self.incept = torch.load(os.path.join(tpath, 'inception_v3.model'))
    else:
        self.incept = torchvision.models.inception_v3(pretrained=True)

    self.incept.eval()


def inceptionv3_init_gpu(self, _):
    import os

    import torch
    import torchvision

    tpath = os.path.join(os.getenv('TORCH_HOME'), 'checkpoints')
    if 'inception_v3.model' in os.listdir(tpath):
        self.incept = torch.load(os.path.join(tpath, 'inception_v3.model'))
    else:
        self.incept = torchvision.models.inception_v3(pretrained=True)

    self.incept = self.incept.cuda()
    self.incept.eval()


def inceptionv3_model(self, img_row: bytes) -> (int, float):
    import torch

    img_t = torch.from_numpy(img_row['img'])
    batch_t = torch.unsqueeze(img_t, 0)
    out = self.incept(batch_t)
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    p_2 = percentage.detach().numpy()

    return indices.detach().numpy()[0][0].item(), \
        p_2[indices[0][0]].item()



def inceptionv3_model_gpu(self, table: Table) -> (int, float):
    import torch

    # Shortcut for empty input.
    if table.size() == 0:
        return []

    originals = [row['img'] for row in table.get()]
    inputs = [torch.from_numpy(img) for img in originals]
    inputs = torch.stack(inputs, dim=0).cuda()

    out = self.incept(inputs)
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    p_2 = percentage.cpu().detach().numpy()

    result = []
    for i in range(len(originals)):
        index = indices[i][0].item()
        perc = p_2[indices[i][0]].item()

        result.append([index, perc])

    return result


def cascade_init(self, cloudburst):
    self.classes = cloudburst.get('imagenet-classes')

def cascade_predict(self, row: Row) -> str:
    resnet_index = row['resnet_index']
    resnet_max_prob = row['resnet_max_prob']
    incept_index = row['incept_index']
    incept_max_prob = row['incept_max_prob']

    if incept_max_prob is None:
        # Didn't go to inception because resnet prediction was confident
        # enough.
        return self.classes[resnet_index]
    else:
        # choose the distribution with the higher max_prob.
        if resnet_max_prob > incept_max_prob:
            return self.classes[resnet_index]
        else:
            return self.classes[incept_index]

def cascade_predict_batch(self, table: Table) -> str:
    results = []
    for row in table.get():
        resnet_index = row['resnet_index']
        resnet_max_prob = row['resnet_max_prob']
        incept_index = row['incept_index']
        incept_max_prob = row['incept_max_prob']

        if incept_max_prob is None:
            # Didn't go to inception because resnet prediction was confident
            # enough.
            results.append(self.classes[resnet_index])
        else:
            # choose the distribution with the higher max_prob.
            if resnet_max_prob > incept_max_prob:
                results.append(self.classes[resnet_index])
            else:
                results.append(self.classes[incept_index])

    return results

def low_prob(self, row: Row) -> bool:
    return row['resnet_max_prob'] < 85

def run(flow, cloudburst, requests, local, sckt=None):
    if not local:
        if not os.path.exists('imagenet_sample.zip'):
            raise RuntimeError('Expect to have the imagenet_sample directory locally.')

        os.system('unzip imagenet_sample.zip')
    else:
        if not os.path.exists('imagenet_sample/imagenet'):
            raise RuntimeError('Expect to have the imagenet_sample directory locally.')


    prefix = 'imagenet_sample/imagenet'
    files = os.listdir(prefix)
    files = [os.path.join(prefix, fname) for fname in files]

    inputs = []

    logging.info('Loading input images...')
    for fname in files:
        table = Table([('img', NumpyType)])
        img = np.array(Image.open(fname).convert('RGB').resize((224, 224)))

        table.insert([img])
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
        resnet = resnet_model_gpu
        resnet_cons = resnet_init_gpu
        incept = inceptionv3_model_gpu
        incept_cons = inceptionv3_init_gpu
        trans = transform_batch
    else:
        resnet = resnet_model
        resnet_cons = resnet_init
        incept = inceptionv3_model
        incept_cons = inceptionv3_init
        trans = transform

    with open('imagenet_classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    cloudburst.put_object('imagenet-classes', classes)

    flow = Flow('cascade-flow', FlowType.PUSH, cloudburst)
    rnet = flow.map(trans,
                    init=transform_init,
                    names=['img'],
                    batching=gpu) \
        .map(resnet,
             init=resnet_cons,
             names=['img', 'resnet_index', 'resnet_max_prob'],
             gpu=gpu,
             batching=gpu)

    incept = rnet.filter(low_prob) \
        .map(incept,
             init=incept_cons,
             names=['incept_index', 'incept_max_prob'],
             gpu=gpu,
             batching=gpu)
    rnet.join(incept, how='left') \
        .map(cascade_predict, init=cascade_init, names=['class'],
             batching=False)


    print('Optimizing flow...')
    flow = optimize(flow, rules=optimize_rules)

    print('Deploying flow...')
    flow.deploy()

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

        sample_input = Table([('img', NumpyType)])
        img = np.array(Image.open('panda.jpg').convert('RGB').resize((224,
                                                                      224)))
        sample_input.insert([img])

        run_distributed_benchmark(flow, args.requests[0], 'cascade',
                                  args.threads[0], benchmark_ips, sample_input)
