from azureml.core import Workspace
import mlflow
from mlflow.tracking import MlflowClient
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import deepspeed


deepspeed.init_distributed()
ws = Workspace.from_config()

if torch.distributed.get_rank() == 0:
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    mlflow.create_experiment("deepspeed-resnet-distributed")
    mlflow.set_experiment("deepspeed-resnet-distributed")
    mlflow_run = mlflow.start_run()

ds_config = {
  "train_batch_size": 16,
  "steps_per_print": 2000,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001,
      "betas": [
        0.8,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 0.001,
      "warmup_num_steps": 1000
    }
  },
  "gradient_clipping": 1.0,
  "prescale_gradients": False,
  "fp16": {
      "enabled": True,
      "fp16_master_weights_and_grads": False,
      "loss_scale": 0,
      "loss_scale_window": 500,
      "hysteresis": 2,
      "min_loss_scale": 1,
      "initial_scale_power": 15
  },
  "wall_clock_breakdown": False,
  "zero_optimization": {
      "stage": 0,
      "allgather_partitions": True,
      "reduce_scatter": True,
      "allgather_bucket_size": 50000000,
      "reduce_bucket_size": 50000000,
      "overlap_comm": True,
      "contiguous_gradients": True,
      "cpu_offload": False
  }
}


#load and prepare the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if torch.distributed.get_rank() != 0:
    # might be downloading cifar data, let rank 0 download first
    torch.distributed.barrier()

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)

if torch.distributed.get_rank() == 0:
    # cifar data is downloaded, indicate other ranks can proceed
    torch.distributed.barrier()

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=ds_config['train_batch_size'],
                                          shuffle=True,
                                          num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=ds_config['train_batch_size'],
                                         shuffle=False,
                                         num_workers=2)


parser = argparse.ArgumentParser(description='CIFAR')

parser.add_argument('--with_cuda',
                    default=True,
                    action='store_true',
                    help='use CPU in case there\'s no GPU support')

parser.add_argument('--use_ema',
                    default=False,
                    action='store_true',
                    help='whether use exponential moving average')

parser.add_argument('-e',
                    '--epochs',
                    default=10,
                    type=int,
                    help='number of total epochs (default: 30)')

parser.add_argument('--local_rank',
                    type=int,
                    default=-1,
                    help='local rank passed from distributed launcher')

parser.add_argument('--log-interval',
                    type=int,
                    default=500,
                    help="output logging information at a given interval")


parser.add_argument('--ep-world-size',
                    default=10,
                    type=int,
                    help='(moe) expert parallel world size')

parser.add_argument('--num-experts',
                    default=40,
                    type=int,
                    help='(moe) number of total experts')

parser.add_argument('--top-k',
                    default=1,
                    type=int,
                    help='(moe) gating top 1 and 2 supported')

parser.add_argument('--min-capacity',
                    default=0,
                    type=int,
                    help='(moe) minimum capacity of an expert regardless of the capacity_factor')

parser.add_argument(
                '--noisy-gate-policy',
                default=None,
                type=str,
                help='(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter')

parser.add_argument(
                '--moe-param-group',
                default=False,
                action='store_true',
                help='(moe) create separate moe param groups, required when using ZeRO w. MoE')

# Include DeepSpeed configuration arguments
parser = deepspeed.add_config_arguments(parser)

args = parser.parse_args("")


mlflow.log_param("ema", args.use_ema)
mlflow.log_param("epochs", args.epochs)
mlflow.log_param("experts", args.num_experts)
mlflow.log_param("noisy gate", args.noisy_gate_policy)

deepspeed.utils.groups.initialize(ep_size=args.ep_world_size)

for i in enumerate(trainloader):
    break
channels = i[1][0].shape[1]
height = i[1][0].shape[-1]
hidden_size = channels * height * height
i[1][0].shape

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,kernel_size=3,padding=1,bias=False):
        super(ResidualBlock,self).__init__()
        self.cnn1 =nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size,1,padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
            
    def forward(self,x):
        residual = x
        x = self.cnn1(x)
        x = self.cnn2(x)
        x += self.shortcut(residual)
        x = nn.ReLU(True)(x)
        return x

class ResNet34(nn.Module):
    def __init__(self, hidden_size, channels, height):
        super(ResNet34,self).__init__()
        self.hidden_size = hidden_size
        self.channels = channels
        self.height = height
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=2,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.block2 = nn.Sequential(
            nn.MaxPool2d(1,1),
            ResidualBlock(64,64),
            ResidualBlock(64,64,2)
        )
        
        self.block3 = nn.Sequential(
            ResidualBlock(64,128),
            ResidualBlock(128,128,2)
        )
        
        self.block4 = nn.Sequential(
            ResidualBlock(128,256),
            ResidualBlock(256,256,2)
        )
        self.block5 = nn.Sequential(
            ResidualBlock(256,512),
            ResidualBlock(512,512,2)
        )
        
        self.avgpool = nn.AvgPool2d(2)

        self.fc1 = nn.Linear(512,self.hidden_size)

    def forward(self,x):
        x = torch.reshape(x, (-1, self.channels, self.height, self.height))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = ResNet34(hidden_size, channels, height)
        self.moe = deepspeed.moe.layer.MoE(
            hidden_size=hidden_size,
            expert=self.resnet,
            num_experts=args.num_experts,
            k=args.top_k,
            min_capacity=args.min_capacity,
            noisy_gate_policy=args.noisy_gate_policy)
        self.fc = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.reshape(-1, hidden_size)
        x, gate_loss, _ = self.moe(x)
        x = self.fc(x)
        return x , gate_loss


net = Net()


def create_moe_param_groups(model):
    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

    parameters = {'params': model.parameters(), 'name': 'parameters'}

    return split_params_into_different_moe_groups_for_optimizer(parameters)


parameters = filter(lambda p: p.requires_grad, net.parameters())
if args.moe_param_group:
    parameters = create_moe_param_groups(net)

# Initialize DeepSpeed to use the following features
# 1) Distributed model
# 2) Distributed data loader
# 3) DeepSpeed optimizer
model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=net, model_parameters=parameters, training_data=trainset, config=ds_config)

fp16 = model_engine.fp16_enabled()
print(f'fp16={fp16}')

import torch.optim as optim

criterion = nn.CrossEntropyLoss()


for epoch in range(args.epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(model_engine.local_rank), data[1].to(
            model_engine.local_rank)
        if fp16:
            inputs = inputs.half()
        outputs, gate_loss = model_engine(inputs)
        loss = criterion(outputs, labels)

        model_engine.backward(loss)
        model_engine.step()

        # print statistics
        running_loss += loss.item()
        if torch.distributed.get_rank() == 0:
            if i % args.log_interval == (args.log_interval - 1):  # print every log_interval mini-batches
                print('training loss [%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / args.log_interval))
                mlflow.log_metric("train loss", running_loss / args.log_interval)
                mlflow.log_metric("gate loss", gate_loss.detach().cpu().numpy() / args.log_interval)
                running_loss = 0.0

print('Finished Training')


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if fp16:
            images = images.half()
        outputs,gate_loss = net(images.to(model_engine.local_rank))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(
            model_engine.local_rank)).sum().item()
if torch.distributed.get_rank() == 0:
    mlflow.log_metric("test accuracy", 100*correct/total)

    print('Accuracy of the network on the 10000 test images: %d %%' %
        (100 * correct / total))

########################################################################
# That looks way better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if fp16:
            images = images.half()
        outputs,gate_loss = net(images.to(model_engine.local_rank))
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels.to(model_engine.local_rank)).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' %
          (trainset.classes[i], 100 * class_correct[i] / class_total[i]))

if torch.distributed.get_rank() == 0:
    mlflow.end_run()

