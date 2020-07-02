import argparse
import logging
import os
import random
import sys
import time
import uuid

from anna.lattices import LWWPairLattice
import cv2
from cloudburst.server.benchmarks.utils import print_latency_stats
from cloudburst.client.client import CloudburstConnection
import cloudpickle as cp
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from flow.benchmarks.utils import run_distributed_benchmark
from flow.operators.flow import Flow, FlowType
from flow.optimize import optimize
from flow.types.basic import NumpyType
from flow.types.table import Row, serialize, Table

optimize_rules = {
    'fusion': False,
    'compete': False,
    'compete_replicas': 0,
    'colocate': False,
    'breakpoint': False,
    'whole': True
}

def yolov3_init(self, cloudburst):
    from torch_yolo.models import Darknet

    self.confidence = 0.8
    self.nms = 0.4

    self.model = Darknet(
        '/usr/local/lib/python3.6/dist-packages/torch_yolo/config/yolov3.cfg',
        img_size=224)
    self.model.load_darknet_weights('/yolo-v3.weights')

    with open('/yolo-v3.classes', 'r') as f:
        self.classes = [line.strip() for line in f.readlines()]

    self.model.eval()

def yolov3_init_gpu(self, cloudburst):
    from torch_yolo.models import Darknet

    self.confidence = 0.8
    self.nms = 0.4

    self.model = Darknet(
        '/usr/local/lib/python3.6/dist-packages/torch_yolo/config/yolov3.cfg',
        img_size=224)
    self.model.load_darknet_weights('/yolo-v3.weights')
    self.model.cuda()

    with open('/yolo-v3.classes', 'r') as f:
        self.classes = [line.strip() for line in f.readlines()]

    self.model.eval()

def yolov3(self, table: Table) -> (np.ndarray, str):
    import torch
    from torch_yolo.utils.utils import non_max_suppression

    originals = [row['frame'] for row in table.get()]
    inputs = [torch.from_numpy(img) for img in originals]
    inputs = torch.stack(inputs, dim=0)

    with torch.no_grad():
        detections = self.model(inputs)
        detections = non_max_suppression(detections, self.confidence, self.nms)

    result = []
    for idx in range(table.size()):
        img = originals[idx].astype(np.uint8) # Convert to int for image models.

        if detections[idx] != None:
            cls = self.classes[int(detections[idx][torch.argmax(detections[idx],
                                                                dim=0)[5]][6])]
        else:
            cls = 'NONE'
        result.append([img, cls])

    return result

def yolov3_gpu(self, table: Table) -> (np.ndarray, str):
    import torch
    from torch_yolo.utils.utils import non_max_suppression

    originals = [row['frame'] for row in table.get()]
    inputs = [torch.from_numpy(img) for img in originals]
    inputs = torch.stack(inputs, dim=0).cuda()

    with torch.no_grad():
        detections = self.model(inputs)
        detections = non_max_suppression(detections, self.confidence, self.nms)

    result = []

    for idx in range(table.size()):
        img = originals[idx].astype(np.uint8) # Convert to int for image models.

        if detections[idx] != None:
            cls = self.classes[int(detections[idx][torch.argmax(detections[idx],
                                                                dim=0)[5]][6])]
        else:
            cls = 'NONE'
        result.append([img, cls])

    return result

def transform_init(self, _):
    from torchvision import transforms

    self.transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

def transform(self, table: Table) -> (np.ndarray, str):
    result = []

    for row in table.get():
        tfmed = self.transform(row['frame']).data.numpy()
        result.append([tfmed, row['class']])

    return result

def resnet_init(self, cloudburst):
    import os

    import torch
    import torchvision

    tpath = os.path.join(os.getenv('TORCH_HOME'), 'checkpoints')
    if 'resnet101.model' in os.listdir(tpath):
        self.resnet = torch.load(os.path.join(tpath, 'resnet101.model'))
    else:
        self.resnet = torchvision.models.resnet101(pretrained=True)

    self.resnet.eval()

    self.classes = cloudburst.get('imagenet-classes')


def resnet_init_gpu(self, cloudburst): # args are self and cloudburst user library
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

    self.classes = cloudburst.get('imagenet-classes')


def resnet_model(self, table: Table) -> str:
    """
    ResNet101 for image classification on ResNet
    """
    import torch

    if table.size() == 0:
        return []

    originals = [row['frame'] for row in table.get()]
    inputs = [torch.from_numpy(img) for img in originals]
    inputs = torch.stack(inputs, dim=0)

    out = self.resnet(inputs)
    _, indices = torch.sort(out, descending=True)
    indices = indices.cpu().detach().numpy()

    result = []
    for i in range(len(originals)):
        index = indices[i][0].item()
        result.append(self.classes[index])

    return result

def resnet_model_gpu(self, table: Table) -> str:
    """
    ResNet101 for image classification on ResNet
    """
    import torch

    if table.size() == 0:
        return []

    originals = [row['frame'] for row in table.get()]
    inputs = [torch.from_numpy(img) for img in originals]
    inputs = torch.stack(inputs, dim=0).cuda()

    out = self.resnet(inputs)
    _, indices = torch.sort(out, descending=True)
    indices = indices.cpu().detach().numpy()

    result = []
    for i in range(len(originals)):
        index = indices[i][0].item()
        result.append(self.classes[index])

    return result


def people_filter(self, row: Row) -> bool:
    return row['class'] == 'person'


def cars_filter(self, row: Row) -> bool:
    return row['class'] == 'car'


def run(flow, cloudburst, requests, local, sckt=None):
    if not local:
        if not os.path.exists('video_sample.zip'):
            raise RuntimeError('Expect to have the video_sample directory locally.')

        os.system('unzip video_sample.zip')

    if not os.path.exists('video_sample/'):
        raise RuntimeError('Expect to have the video_sample directory locally.')

    prefix = 'video_sample'
    files = os.listdir(prefix)
    files = [os.path.join(prefix, fname) for fname in files]

    inputs = []

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((720, 720)),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    import time
    logging.info('Loading input videos...')
    for fname in files:
        table = Table([('frame', NumpyType)])

        cap = cv2.VideoCapture(fname)

        more_frames, frame = cap.read()
        while more_frames and table.size() < 30:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = transform(frame).detach().numpy()
            table.insert([frame])
            more_frames, frame = cap.read()

        key = str(uuid.uuid4())
        lattice = LWWPairLattice(0, serialize(table))
        cloudburst.kvs_client.put(key, lattice)

        inputs.append(key)

    logging.info('Starting benchmark...')

    latencies = []
    bench_start = time.time()
    for i in range(requests):
        if i % 1 == 0:
            logging.info(f'On request {i}...')

        inp = random.choice(inputs)
        start = time.time()
        fut = flow.run(inp)
        result = fut.get()
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
        yolo = yolov3_gpu
        yolo_cons = yolov3_init_gpu
    else:
        resnet = resnet_model
        resnet_cons = resnet_init
        yolo = yolov3
        yolo_cons = yolov3_init

    with open('imagenet_classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    cloudburst.put_object('imagenet-classes', classes)

    flow = Flow('video-flow', FlowType.PUSH, cloudburst)
    imgs = flow.map(yolo,
                    init=yolo_cons,
                    names=['frame', 'class'],
                    gpu=gpu,
                    multi=True
                    ) \
        .map(transform,
             init=transform_init,
             names=['frame', 'class'],
             multi=True
             )

    people = imgs.filter(people_filter) \
        .map(resnet,
             init=resnet_cons,
             names=['people-class'],
             gpu=gpu,
             multi=True
             )
    cars = imgs.filter(cars_filter) \
        .map(resnet,
             init=resnet_cons,
             names=['vehicle-class'],
             gpu=gpu,
             multi=True
             )

    people.join(cars, how='outer')

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

        sample_input = Table([('frame', NumpyType)])
        cap = cv2.VideoCapture('video_sample/steph3.mp4')

        img_tfms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((720, 720)),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        more_frames, frame = cap.read()
        while more_frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = img_tfms(frame).detach().numpy()
            sample_input.insert([frame])
            more_frames, frame = cap.read()

        run_distributed_benchmark(flow, args.requests[0], 'video',
                                  args.threads[0], benchmark_ips, sample_input)
