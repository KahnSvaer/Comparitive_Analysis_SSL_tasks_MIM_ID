SimCLR_model_checkpoint = "/kaggle/input/simclr-model-100/runs/May27_10-38-58_a9089d95001b/checkpoint_epoch_0100.pth.tar"
LinearProbe_checkpoint = '/kaggle/input/linearprobing-simclr-linhead-40-with-eval/models/best_model.pth.tar'
Validation_dataset_path = "/kaggle/input/ssl-dataset/ssl_dataset/val.X"
Class_idx_json_path = '/kaggle/input/linearprobing-simclr-featureextractor-valid/class_to_idx.json'



import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms
import torchvision.models as models
from torchvision import datasets

# 
import os

import yaml

import re

torch.manual_seed(0)
np.random.seed(0)


class BaseSimCLRException(Exception):
    """Base exception"""


class InvalidBackboneError(BaseSimCLRException):
    """Raised when the choice of backbone Convnet is invalid."""


class InvalidDatasetSelection(BaseSimCLRException):
    """Raised when the choice of dataset is invalid."""


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name = 'resnet50'):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# In[109]:


model = ResNetSimCLR(base_model="resnet50", out_dim=128)


if torch.cuda.is_available():
    device = torch.device("cuda")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
else:
    device = torch.device("cpu")

saved_checkpoint = torch.load(SimCLR_model_checkpoint, map_location=device)
msg = model.load_state_dict(saved_checkpoint['state_dict'])
print(msg)


# In[110]:


# Loading in the Linear Probe Model

LinearProbe = nn.Sequential(
        nn.BatchNorm1d(2048, affine=False, eps=1e-6),
        nn.Linear(2048, 100)
    ).to(device)

saved_checkpoint = torch.load(LinearProbe_checkpoint, map_location=device)
msg = LinearProbe.load_state_dict(saved_checkpoint['model_state_dict'])
print(msg)


# In[111]:


encoder = model.module.backbone
encoder.fc = nn.Identity()
encoder.to(device)
if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        encoder = torch.nn.DataParallel(encoder)


# In[119]:


from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=Validation_dataset_path, transform=transform)

with open(Class_idx_json_path, 'r') as f:
    class_to_idx = json.load(f)

dataset.class_to_idx = class_to_idx

dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)


# In[120]:


print(saved_checkpoint)


# In[121]:


import torch

encoder.eval()
LinearProbe.eval()

correct = 0
total = 0

with torch.no_grad():
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        features = encoder(batch_X)
        logits = LinearProbe(features)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

accuracy = correct / total
print(f"Evaluation Accuracy: {accuracy:.4f}")


# In[ ]:


# Not sure where the error is the same process gave 0.46 last time

