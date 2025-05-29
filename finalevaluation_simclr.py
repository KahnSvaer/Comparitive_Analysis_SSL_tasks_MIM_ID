import numpy as np
import torch
from torch import nn
import torchvision.models as models
from torchvision import datasets
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import argparse

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




def main():

    parser = argparse.ArgumentParser(description='Evaluate linear probe on SSL features')

    parser.add_argument('--checkpoint', type=str, default='/kaggle/working/model/save_model.pth',
                        help='Path to the saved checkpoint (.pth file)')
    parser.add_argument('--val-data', type=str, default='/kaggle/input/ssl-dataset/ssl_dataset/val.X',
                        help='Path to the validation dataset directory')

    args, _ = parser.parse_known_args()

    Saved_Checkpoint_Path = args.checkpoint
    Validation_dataset_path = args.val_data
    Class_idx_json_path = "./class_to_idx_save.json"
                            
    model = ResNetSimCLR(base_model="resnet50", out_dim=128)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
            
    if isinstance(model, nn.DataParallel):
        encoder = model.module.backbone
    else:
        encoder = model.backbone
    encoder.fc = nn.Identity()
    encoder.to(device)
    
    saved_checkpoint = torch.load(Saved_Checkpoint_Path, map_location=device)
    msg = encoder.load_state_dict(saved_checkpoint['encoder_state_dict'])
    print(msg)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            encoder = nn.DataParallel(encoder)
        else:
            device = torch.device("cpu")
    
    
    LinearProbe = nn.Sequential(
            nn.BatchNorm1d(2048, affine=False, eps=1e-6),
            nn.Linear(2048, 100)
        ).to(device)
    
    msg = LinearProbe.load_state_dict(saved_checkpoint['linear_state_dict'])
    print(msg)
    
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(root=Validation_dataset_path, transform=transform)
    
    with open(Class_idx_json_path, 'r') as f:
        class_to_idx = json.load(f)
    
    dataset.class_to_idx = class_to_idx
    
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    
    
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

if __name__ == "__main__":
    main()

# Call this file by running  python finalevaluation_simclr.py --checkpoint path/to/checkpoint.pth --val-data path/to/val_data

# Not sure where the error is the same process gave 0.46 on notebook but this results to 0.41. Possible due to shu
