import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

from vit_pytorch import ViT
import torch.optim as optim
import torch.nn as nn
import datetime
import numpy as np
import argparse


# train the VIT model with parrallel GPU
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=True, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)

# make the following code into a function to be called in the main function
def train(optimizer: str, epochs: int, lr: float, batch_size: int, 
            img_size: int, num_workers: int, device: str, data_train_dir: str, data_test_dir: str,
            save_dir: str, model_name: str, patch_size: int, num_classes: int,
            dim: int, depth: int, heads: int, mlp_dim: int, dropout: float,
            emb_dropout: float, no_report_wandb: bool, no_save_model: bool, cosine_lr: bool, 
            early_stop: bool, project_name: str, run_name: str, wd: float, rho: float):
    
    optimizer_name = optimizer
    # set up wandb if not disabled
    if not no_report_wandb:
        import wandb
        # initialize wandb
        wandb.init(project=project_name, name=run_name)

        # save wandb config
        wandb.config.update({
            "optimizer": optimizer_name,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "img_size": img_size,
            "num_workers": num_workers,
            "device": device,
            "data_train_dir": data_train_dir,
            "data_test_dir": data_test_dir,
            "save_dir": save_dir,
            "model_name": model_name,
            "patch_size": patch_size,
            "num_classes": num_classes,
            "dim": dim,
            "depth": depth,
            "heads": heads,
            "mlp_dim": mlp_dim,
            "dropout": dropout,
            "emb_dropout": emb_dropout,
            "no_report_wandb": no_report_wandb,
            "no_save_model": no_save_model,
            "cosine_lr": cosine_lr,
            "early_stop": early_stop,
        })
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    # Load the ImageNet100 dataset
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size+32),
        transforms.CenterCrop(img_size),
        normalize,
    ])

    trainset = ImageFolder(root=data_train_dir, transform=transform_train)
    testset = ImageFolder(root=data_test_dir, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                num_workers=num_workers, drop_last=True, pin_memory=True)

                        
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, drop_last=False, pin_memory=True)

    # move device to GPU if available
    try:
        device = torch.device(device)
    except:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # print device to wandb
    if not no_report_wandb:
        wandb.config.device = device

    # Set up a simple vision transformer model
    model = ViT(
        image_size = img_size,
        patch_size = patch_size,
        num_classes = num_classes,
        dim = dim,
        depth = depth,
        heads = heads,
        mlp_dim = mlp_dim,
        dropout = dropout,
        emb_dropout = emb_dropout
    ).to(device)
    
    
    # initialize SAM optimizer
    if optimizer_name[:3] == "SAM":
        # params { rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay
        sam_params = {
            "rho": rho,
            "adaptive": True,
            "lr": lr,
            "weight_decay": wd,
        }

        if optimizer_name == 'SAM-AdamW':
            base_optimizer = torch.optim.AdamW
            optimizer = SAM(model.parameters(), base_optimizer, **sam_params)
        elif optimizer_name == 'SAM-Adam':
            base_optimizer = torch.optim.Adam
            optimizer = SAM(model.parameters(), base_optimizer, **sam_params)
        elif optimizer_name == 'SAM-SGD':
            base_optimizer = torch.optim.SGD
            optimizer = SAM(model.parameters(), base_optimizer, **sam_params)
    else:
        criterion = nn.CrossEntropyLoss()
        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        elif optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            print("Optimizer not supported, using SGD")
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # set up cosine learning rate scheduler
    if cosine_lr:
        optimizer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0, last_epoch=-1)


    # print model to wandb
    if not no_report_wandb:
        wandb.watch(model)

    try:
        label_smoothing = 0.1
        for epoch in range(epochs):  # loop over the dataset multiple times
            # track performance at each epoch
            valid_performance = []
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # if SAM optimizer, use first order gradient
                if optimizer_name == 'SAM-AdamW' or optimizer_name == 'SAM-Adam' or optimizer_name == 'SAM-SGD':
                    # first forward-backward step
                    # enable_running_stats(model)
                    enable_running_stats(model)
                    predictions = model(inputs)
                    loss = smooth_crossentropy(predictions, labels, smoothing=label_smoothing)
                    loss.mean().backward()
                    optimizer.first_step(zero_grad=True)

                    # second forward-backward step
                    disable_running_stats(model)
                    smooth_crossentropy(model(inputs), labels, smoothing=label_smoothing).mean().backward()
                    optimizer.second_step(zero_grad=True)
                    with torch.no_grad():
                        correct = torch.argmax(predictions.data, 1) == labels

                else:
                    # forward + backward + optimize
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                

                # print statistics
                running_loss += loss.mean().item()
                if i % 200 == 199:    # print every 2000 mini-batches
                    if not no_report_wandb:
                        wandb.log({'loss': running_loss / 200})
                    running_loss = 0.0

            # Test the VIT model
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    outputs = model(images.to(device))
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels.to(device)).sum().item()
            
            # Training accuracy
            correct_train = 0
            total_train = 0
            with torch.no_grad():
                for data in trainloader:
                    images, labels = data
                    outputs = model(images.to(device))
                    _, predicted = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels.to(device)).sum().item()

            # print accuracy to wandb
            if not no_report_wandb:
                wandb.log({"Test accuracy": correct_test / total_test})
                wandb.log({"Train accuracy": correct_train / total_train})
            print("Epoch: ", epoch, "Test Accuracy: ", correct_test / total_test, "Train Accuracy: ", correct_train / total_train)

            valid_performance.append(correct_test / total_test)

            if optimizer == 'SAM-AdamW' or optimizer == 'SAM-Adam' or optimizer == 'SAM-SGD':
                optimizer.update_learning_rate(valid_performance)
                # report learning rate update to wandb
                if not no_report_wandb:
                    wandb.log({'lr': optimizer.param_groups[0]['lr']})

            # if performance is not improving for 50 epochs, stop training
            # using the mean of 5 epochs
            if len(valid_performance) > early_stop:
                if np.mean(valid_performance[-3:]) < np.mean(valid_performance[-early_stop:-early_stop+3]):
                    print("Performance not improving, stopping training")
                    break
    except Exception as e: 
        # print out exception 
        print(e)
        print("\n\nTraining interrupted")
        print("Saving model and ending training...")
        
    if not no_save_model: 
        print("Saving model...")
        
        # save the model to wandb
        if not no_report_wandb:
            wandb.save(model_name+".pt")

        # save the model with date and time
        torch.save(model.state_dict(), save_dir+'/'+model_name+".pt")
        # save model training to wandb
        if not no_report_wandb:
            wandb.finish()


# call train function with arg parser for epochs, lr, batch size, etc.
if __name__ == "__main__":
    # find current date and time for saving model 
    now = datetime.datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")

    parser = argparse.ArgumentParser(description='Train ViT model')
     
    parser.add_argument('--optimizer', type=str, default='SAM-AdamW', help='optimizer to use')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--img_size', type=int, default=256, help='image size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use')
    parser.add_argument('--data_train_dir', type=str, default='./data/imagenet100/train', help='path to train data')
    parser.add_argument('--data_test_dir', type=str, default='./data/imagenet100/val', help='path to test data')
    parser.add_argument('--save_dir', type=str, default='./models', help='path to save model')
    parser.add_argument('--model_name', type=str, default='transformer'+date_time, help='name of model')
    parser.add_argument('--patch_size', type=int, default=32, help='patch size')
    parser.add_argument('--num_classes', type=int, default=1000, help='number of classes')
    parser.add_argument('--dim', type=int, default=1024, help='dim')
    parser.add_argument('--depth', type=int, default=6, help='depth')
    parser.add_argument('--heads', type=int, default=16, help='heads')
    parser.add_argument('--mlp_dim', type=int, default=2048, help='mlp_dim')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--emb_dropout', type=float, default=0.1, help='emb_dropout')
    parser.add_argument('--no_report_wandb', action='store_true', help='report to wandb')
    parser.add_argument('--no_save_model', action='store_true', help='save model')
    parser.add_argument('--cosine_lr', action='store_true', help='use cosine learning rate')
    parser.add_argument('--early_stop', type=int, default=20, help='Number of epochs with no improvement before early stopping')
    parser.add_argument('--project_name', type=str, default='hyperparameter_optimization', help='wandb project name')
    parser.add_argument('--run_name', type=str, default='transformer'+date_time, help='wandb run name')
    parser.add_argument('--wd', type=float, default=0.3, help='weight decay')
    parser.add_argument('--rho', type=float, default=0.05, help='SAM rho')

    args = parser.parse_args()

    # call train function
    train(**vars(args))


