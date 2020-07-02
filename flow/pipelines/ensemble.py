import os

import numpy as np

from flow.operators.flow import Flow, FlowType
from flow.types.basic import BoolType, BtsType, FloatType, IntType, StrType
from flow.types.table import Row, Table

import torch, torchvision

def transform_init(self, _):
    from torchvision import transforms
    self.transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

def transform(self, img_row: bytes) -> np.ndarray:
    from PIL import Image
    import base64
    from io import BytesIO

    img = Image.open(BytesIO(base64.b64decode(img_row['img']))).convert('RGB').resize((224, 224))

    return self.transform(img).detach().numpy()

def alexnet_init(self, _): # args are self and cloudburst user library
    import os

    import torch
    import torchvision

    tpath = os.path.join(os.getenv('TORCH_HOME'), 'checkpoints')
    if 'alexnet.model' in os.listdir(tpath):
        self.alexnet = torch.load(os.path.join(tpath, 'alexnet.model'))
    else:
        self.alexnet = torchvision.models.alexnet(pretrained=True)

    self.alexnet.eval()

def alexnet_model(self, img_row: Row) -> (np.ndarray, np.ndarray):
    """
    AlexNet for image classification on ImageNet
    """
    import torch

    img_t = torch.from_numpy(img_row['img'])
    batch_t = torch.unsqueeze(img_t, 0)
    out = self.alexnet(batch_t)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)

    return indices.detach().numpy()[0], percentage.detach().numpy()

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

def resnet_model(self, img_row: Row) -> (np.ndarray, np.ndarray):
    """
    ResNet101 for image classification on ResNet
    """
    import torch

    img_t = torch.from_numpy(img_row['img'])
    batch_t = torch.unsqueeze(img_t, 0)
    out = self.resnet(batch_t)
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    return indices.detach().numpy()[0], percentage.detach().numpy()

def ensemble_predict(self, predict_row: (int, float)) -> str:
    """
    Ensembling (via average) models for image classification
    """
    with open('/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    import numpy as np

    a_index = predict_row['alexnet_index']
    a_perc = predict_row['alexnet_perc']
    r_index = predict_row['resnet_index']
    r_perc = predict_row['resnet_perc']
    all_percentages = (a_perc + r_perc) / 2
    indices = np.argsort(all_percentages)[::-1]
    return classes[indices[0]]


import base64
import sys

from cloudburst.client.client import CloudburstConnection

table = Table([('img', StrType)])
img = base64.b64encode(open('panda.jpg', "rb").read()).decode('ascii')

table.insert([img])

cloudburst = CloudburstConnection(sys.argv[1], '3.226.122.35')
flow = Flow('ensemble-flow', FlowType.PUSH, cloudburst)
img = flow.map(transform, init=transform_init, names=['img'])

anet = img.map(alexnet_model, init=alexnet_init, names=['alexnet_index', 'alexnet_perc'])
rnet = img.map(resnet_model, init=resnet_init, names=['resnet_index', 'resnet_perc'])
anet.join(rnet).map(ensemble_predict, names=['class'])

flow.deploy()

from cloudburst.server.benchmarks.utils import print_latency_stats
import time

print('Starting benchmark...')

latencies = []
for _ in range(100):
    start = time.time()
    result = flow.run(table).get()
    end = time.time()
    time.sleep(1)

    latencies.append(end - start)

print_latency_stats(latencies, "E2E")
