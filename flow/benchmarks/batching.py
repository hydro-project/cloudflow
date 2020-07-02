import argparse
import os
import sys
import time

from cloudburst.client.client import CloudburstConnection
from cloudburst.server.benchmarks.utils import print_latency_stats
import numpy as np
from PIL import Image

from flow.operators.flow import Flow, FlowType
from flow.types.basic import NumpyType
from flow.types.table import Row, Table

def run(cloudburst: CloudburstConnection,
        num_requests: int,
        batch_size: int,
        gpu: bool):

    with open('imagenet_classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    cloudburst.put_object('imagenet-classes', classes)

    def resnet_init_gpu(self, cloudburst):
        import os

        import torch
        import torchvision
        from torchvision import transforms

        tpath = os.path.join(os.getenv('TORCH_HOME'), 'checkpoints')
        self.resnet = torch.load(os.path.join(tpath, 'resnet101.model')).cuda()
        self.resnet.eval()

        self.transforms = transforms.Compose(
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

        self.classes = cloudburst.get('imagenet-classes')

    def resnet_model_gpu(self, table: Table) -> str:
        """
        AlexNet for image classification on ImageNet
        """
        import torch

        inputs = []
        for row in table.get():
            img = self.transforms(row['img'])
            inputs.append(img)

        inputs = torch.stack(inputs, dim=0).cuda()
        output = self.resnet(inputs)
        _, indices = torch.sort(output, descending=True)
        indices = indices.cpu().detach().numpy()

        result = []
        for idx_set in indices:
            index = idx_set[0]
            result.append(self.classes[index])

        return result

    def resnet_init_cpu(self, cloudburst):
        import os

        import torch
        import torchvision
        from torchvision import transforms

        tpath = os.path.join(os.getenv('TORCH_HOME'), 'checkpoints')
        self.resnet = torch.load(os.path.join(tpath, 'resnet101.model'))

        self.resnet.eval()

        self.transforms = transforms.Compose(
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

        self.classes = cloudburst.get('imagenet-classes')

    def resnet_model_cpu(self, table: Table) -> str:
        """
        AlexNet for image classification on ImageNet
        """
        import torch

        inputs = []
        for row in table.get():
            img = self.transforms(row['img'])
            inputs.append(img)

        inputs = torch.stack(inputs, dim=0)
        output = self.resnet(inputs)
        _, indices = torch.sort(output, descending=True)
        indices = indices.detach().numpy()

        result = []
        for idx_set in indices:
            index = idx_set[0]
            result.append(self.classes[index])

        return result

    print(f'Creating flow with size {batch_size} batches.')

    flow = Flow('batching-benchmark', FlowType.PUSH, cloudburst)
    if gpu:
        flow.map(resnet_model_gpu, init=resnet_init_gpu, names=['class'],
                 gpu=True, batching=True)
    else:
        flow.map(resnet_model_cpu, init=resnet_init_cpu, names=['class'],
                 batching=True)

    flow.deploy()
    print('Flow successfully deployed!')

    latencies = []
    inp = Table([('img', NumpyType)])
    img = np.array(Image.open('panda.jpg').convert('RGB').resize((224, 224)))

    inp.insert([img])

    kvs = cloudburst.kvs_client

    if gpu:
        print('Starting GPU warmup...')
        for _ in range(50):
            flow.run(inp).get()
        print('Finished warmup...')

    print('Starting benchmark...')
    for i in range(num_requests):
        if i % 100 == 0 and i > 0:
            print(f'On request {i}...')

        futs = []
        for _ in range(batch_size):
            futs.append(flow.run(inp))
        pending = set([fut.obj_id for fut in futs])

        # Break these apart to batch the KVS get requests.
        start = time.time()
        while len(pending) > 0:
            get_start = time.time()
            response = kvs.get(list(pending))

            for key in response:
                if response[key] is not None:
                    pending.discard(key)

        end = time.time()
        latencies.append(end - start)

    compute_time = np.mean(latencies) * num_requests
    tput = (batch_size * num_requests) / (compute_time)
    print('THROUGHPUT: %.2f' % (tput))
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
    parser.add_argument('-b', '--batch', nargs=1, type=int, metavar='S',
                        help='The pre-determined batch size to use (required)',
                        dest='batch', required=True)
    parser.add_argument('-g', '--breakpoint', nargs=1, type=str, metavar='B',
                        help='Whether to enable GPU execution (required)',
                        dest='gpu', required=True)

    args = parser.parse_args()

    cloudburst = CloudburstConnection(args.cloudburst[0], args.ip[0],
                                      local=True)
    print('Successfully connected to Cloudburst')

    run(cloudburst, args.requests[0], args.batch[0], args.gpu[0].lower() == 'true')
