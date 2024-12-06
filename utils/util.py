import models.resnet3d
from models import *
from dataset import *
import torchvision
import torch
from torchvision import transforms
import os
import torch.distributed as dist

def init_comm_size_and_rank():
    world_size = None
    world_rank = 0

    if os.getenv("OMPI_COMM_WORLD_SIZE") and os.getenv("OMPI_COMM_WORLD_RANK"):
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])

    return int(world_size), int(world_rank)

def get_comm_size_and_rank():
    world_size, world_rank = init_comm_size_and_rank()
    return world_size, world_rank

def get_distributed_model(model, local_rank,verbosity=0, sync_batch_norm=False):
    device_name = get_device_name(verbosity_level=verbosity)
    if dist.is_initialized():
        #if device_name == "cpu":
        #    #device = get_device(device_name)
        #    device = get_device_from_name(device_name)
        #    print("RANK, DEVICE NAME:", dist.get_rank(), device, flush=True)
        #    device = torch.device('cuda:{}'.format(local_rank))
        #    model.to(device)
        #    model = torch.nn.parallel.DistributedDataParallel(model)
        #else: 
        if sync_batch_norm:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # device = get_device()
        device = get_device_from_name(device_name)
        # print("RANK, local_rank, DEVICE NAME:", dist.get_rank(), local_rank,device, flush=True)
        device = torch.device('cuda:{}'.format(local_rank))
        model.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
            #model, device_ids=[local_rank], output_device=[local_rank],process_group=None, find_unused_parameters=True
        )
    return model, device_name

def get_device_list():
    num_devices = torch.cuda.device_count()
    device_list = [torch.cuda.get_device_name(i) for i in range(num_devices)]
    if dist.get_rank() == 0:
        print("DEVICE LIST", device_list, flush=True)
    return device_list

def get_device_name(use_gpu=True, rank_per_model=1, verbosity_level=0, no_prefix=False):

    available_gpus = get_device_list()
    if not use_gpu or not available_gpus:
        # print_distributed(verbosity_level, "Using CPU")
        return "cpu"

    world_size, world_rank = get_comm_size_and_rank()
    if rank_per_model != 1:
        raise ValueError("Exactly 1 rank per device currently supported")

    # print_distributed(verbosity_level, "Using GPU")
    ## We need to ge a local rank if there are multiple GPUs available.
    localrank = 0
    if torch.cuda.device_count() > 1:
        if os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"):
            ## Summit
            localrank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

        if localrank >= torch.cuda.device_count():
            print(
                "WARN: localrank is greater than the available device count - %d %d"
                % (localrank, torch.cuda.device_count())
            )

    if no_prefix:
        device_name = str(localrank)
    else:
        device_name = "cuda:" + str(localrank)
    #print("RANK, DEVICE NAME:", dist.get_rank(), device_name, flush=True)

    return device_name

def get_device_from_name(name: str):
    return torch.device(name)

normalize_list = {'clip': transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                  'general': transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                  'imagenet': transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                  'gray': transforms.Normalize(mean=[0.5], std=[0.5])
                  }

def adjust_learning_rate(learning_rate, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    learning_rate = learning_rate * (0.1 ** (epoch // 30))  # args.lr = 0.1 ,
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def get_surrogate(name, local_rank, num_classes=1000):
    if name == 'RN50':
        model = resnet50(pretrained=False)
        model.load_state_dict(torch.load('checkpoints/RN50_imagenet.pth'))
        # model, device = get_distributed_model(model, local_rank)
        # torch.save(model.state_dict(), 'checkpoints/RN50_imagenet.pth')
    elif name == 'CLIPRN50':
        model = ClipResnet(name='RN50', num_classes=num_classes)
        model.load_pretrain()
        torch.save(model.state_dict(), 'checkpoints/RN50_clip.pth')
    elif name =='RN50_gray':
        model = resnet50(pretrained = True)
        torch.save(model.state_dict(), 'checkpoints/RN50_imagenet.pth')
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif name == 'RN50_3d':
        model = models.resnet3d.resnet503d(pretrained=False)
    else:
        raise f'Model {name} Not Found'
    return model# , device


def get_target(name, local_rank, num_classes=1000):
    if name == 'RN18':
        model = resnet18(num_classes=num_classes)
        model, device = get_distributed_model(model, local_rank)
    elif name == 'RN18_gray':
        model = resnet18(num_classes=num_classes)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model, device = get_distributed_model(model, local_rank)
    elif name == 'SimCLRRN50':
        model = SimCLR(name='RN50', num_classes=num_classes)
    elif name == 'regnet':
        model = torchvision.models.regnet_x_1_6gf(num_classes=num_classes)
    elif name == 'efficientnet_b1':
        model = torchvision.models.efficientnet_b1(num_classes=num_classes)
    elif name == 'RN18_gray':
        model = resnet18(num_classes=num_classes)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model, device = get_distributed_model(model, local_rank)
    else:
        raise (f'Model {name} Not Found')
    return model, device
