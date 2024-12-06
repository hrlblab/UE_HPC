import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import os
from pathlib import Path
from ruamel.yaml import YAML
import argparse
from logger import MetricLogger
from utils import get_surrogate, get_target, normalize_list
from dataset.dataCluster import DataFolderWithLabel, DataFolderWithClassNoise
from models.generator import ResnetGenerator
from models.generator3d import ResnetGenerator3d
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
import subprocess
import horovod.torch as hvd
import datetime
import torchvision.models as models

class ResNetWithFeature(nn.Module):
    def __init__(self):
        super(ResNetWithFeature, self).__init__()
        self.resnet = models.resnet50(pretrained=False)  # 使用 ResNet50
        self.features = {}

        # self.resnet.fc = nn.Identity()

    def forward(self, x):
        # 提取前面几层的特征
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # 全局平均池化
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)  # flatten 作为 fc 层的输入

        # 保存全连接层的输入作为 features['feat']
        self.features['feat'] = x.clone()
        # 全连接层的输出
        output = self.resnet.fc(x)
        return output, x

class ResNetGrayWithFeature(nn.Module):
    def __init__(self):
        super(ResNetGrayWithFeature, self).__init__()
        self.resnet = models.resnet50(pretrained=False)  # 使用 ResNet50
        self.features = {}

        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # self.resnet.fc = nn.Identity()

    def forward(self, x):
        # 提取前面几层的特征
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # 全局平均池化
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)  # flatten 作为 fc 层的输入

        # 保存全连接层的输入作为 features['feat']
        self.features['feat'] = x.clone()
        # 全连接层的输出
        output = self.resnet.fc(x)
        return output, x

def initialize_world():
    num_gpus_per_node = torch.cuda.device_count()
    
    import subprocess
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

    get_master = "echo $(cat {} | sort | uniq | grep -v batch | grep -v login | head -1)".format(os.environ["LSB_DJOB_HOSTFILE"])
    os.environ['MASTER_ADDR'] = str(subprocess.check_output(get_master, shell=True))[2:-3]
    os.environ['MASTER_PORT'] = "23456"

    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(world_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=world_rank,
            world_size=world_size,
            )
    return world_size, world_rank, local_rank

def init_comm_size_and_rank():
    world_size = None
    world_rank = 0

    if os.getenv("OMPI_COMM_WORLD_SIZE") and os.getenv("OMPI_COMM_WORLD_RANK"):
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])

    return int(world_size), int(world_rank)

def get_local_rank():
    local_rank = 0
    if os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"):
        localrank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

    return localrank

#def setup(rank, world_size):
def setup():

    if not dist.is_initialized():
       world_size, world_rank = init_comm_size_and_rank()
       return world_size, world_rank

    world_size, world_rank = init_comm_size_and_rank()

    if os.getenv("LSB_HOSTS") is not None:
        master_addr = os.environ["LSB_HOSTS"].split()[1]

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(world_rank)
    os.environ["LOCAL_RANK"] = str(get_local_rank())

    if dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=1800)
        )
    
    return world_size, world_rank

    ## get_master = "echo $(cat {} | sort | uniq | grep -v batch | grep -v login | head -1)".format(os.environ['LSB_DJOB_HOSTFILE'])
    #try:
    #    result = subprocess.run("cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch", shell=True, capture_output=True, text=True, check=True)
    #    nodes = result.stdout.strip().split('\n')
    #    head = nodes[0] if nodes else None
    #    if head:
    #        os.environ['MASTER_ADDR'] = head
    #        print(f"Setting env_var MASTER_ADDR = {head}")
    #    else:
    #        print("No valid nodes found")

    #except subprocess.CalledProcessError as e:
    #    print(f"Error while getting nodes: {e}")
    ## os.environ['MASTER_ADDR'] = str(subprocess.check_output(get_master, shell=True))[2:-3]
    #os.environ['MASTER_PORT'] = '29500'
    #os.environ['WORLD_SIZE'] = os.environ.get('OMPI_COMM_WORLD_SIZE', '1')
    #os.environ['RANK'] = os.environ.get('OMPI_COMM_WORLD_RANK', '0')
    #os.environ['LOCAL_RANK'] = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')
    #dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

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

    return device_name

def get_device_from_name(name: str):
    return torch.device(name)

def get_device(use_gpu=True, rank_per_model=1, verbosity_level=0):
    name = get_device_name(use_gpu, rank_per_model, verbosity_level=0)
    return get_device_from_name(name)

def get_comm_size_and_rank():
    world_size, world_rank = init_comm_size_and_rank()
    return world_size, world_rank

def get_device_list():
    num_devices = torch.cuda.device_count()
    device_list = [torch.cuda.get_device_name(i) for i in range(num_devices)]
    return device_list

def get_distributed_model(model, local_rank, verbosity=0, sync_batch_norm=False):
    device_name = get_device_name(verbosity_level=verbosity)

    if dist.is_initialized():
        if device_name == "cpu":
            #device = get_device(device_name)
            device = get_device_from_name(device_name)
            model.to(device)
            model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            if sync_batch_norm:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            #device = get_device()
            device = get_device_from_name(device_name)
            model.to(local_rank)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], find_unused_parameters=True
            )
    return model, device_name

def cleanup():
    dist.destroy_process_group()

#def train_gnet(rank, world_size, args, config):
def train_gnet(args, config):
    #setup(rank, world_size)
    world_size, world_rank, local_rank = initialize_world()
    #device = torch.device(f'cuda:{rank}')
    # hvd.init()
    # torch.cuda.set_device(hvd.local_rank())
    # print(torch.cuda.get_device_name(args.device))
    train_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    #print(torch.cuda.get_device_name(args.device))
    normalize = normalize_list[config['normalize']]
    batch_size = config['batch_size']
    local_batch_size = batch_size // world_size + 1
    print("world_size: ", world_size, "BATCH", batch_size, "LOCAL_BATCHSIZE: ", local_batch_size)

    # net = get_surrogate(config['model'], config['num_classes']).eval().to(args.device)
    # sd = torch.load(config['checkpoint'], map_location='cpu')
    # net.load_state_dict(sd)

    # net = get_surrogate(config['model'], config['num_classes']).eval().to(device)
    # net, device = get_surrogate(config['model'], local_rank, config['num_classes'])
    # pretrained_model = models.resnet50(pretrained=True)
    # pretrained_dict = pretrained_model.state_dict()
    
    model = ResNetWithFeature()

    checkpoints = torch.load('checkpoints/RN50_imagenet.pth')
    # model.load_state_dict(torch.load('checkpoints/RN50_imagenet.pth'))
    model_dict = model.state_dict()
    checkpoints = {k: v for k, b in checkpoints.items() if k in model_dict and 'fc' not in k}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    model_dict.update(checkpoints)
    model.load_state_dict(model_dict)
    net, device = get_distributed_model(model, local_rank)
    # net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [rank], find_unused_parameters=True)
    # net,device = get_distributed_model(net)
    
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    cluster = torch.load(config['cluster'], map_location='cpu')
    cluster['centers'] = cluster['centers'].to(local_rank)
    num_clusters = cluster['centers'].shape[0]

    train_dataset = DataFolderWithLabel(config['dataset']['config']['train'], cluster['pred_idx'], train_transform)
    # train_loader = DataLoader(train_dataset, batch_size=256, num_workers=8, pin_memory=True)
    # train_dataset = DataFolderWithLabel(config['dataset']['config']['train'], cluster['pred_idx'], train_transform)
    sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=world_rank)
    # train_loader = DataLoader(train_dataset, batch_size=local_batch_size, num_workers=4, pin_memory=True, sampler=sampler)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, sampler=sampler)
    # Using hvd
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replocas=hvd.size(), rank=hvd.rank())
    # train_loader = torch.utils.data.Dataloader(train_dataset, batch_size=256, num_workers=8, pin_memory=True, sampler=train_sampler)

    for cluster_idx in range(num_clusters):
        noise = torch.zeros((1, 3, 224, 224)) # 修改噪声形状
        noise.uniform_(0, 1)
        noise = noise.to(local_rank)

        g_net = ResnetGenerator(3, 3, 64, norm_type='batch', act_type='relu') # 修改生成网络通道数
        g_net,device = get_distributed_model(g_net, local_rank)
        # g_net.to(args.device)
        #g_net.cuda()

        optimizer = torch.optim.Adam(g_net.parameters(), lr=config['lr'], weight_decay=5e-4)
        # optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.name_parameters())
        # hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['num_epoch'] * len(train_loader), eta_min=1e-6)
        criterion = torch.nn.KLDivLoss(reduction='batchmean')

        # logger = MetricLogger()
        if world_rank == 0:
            print("CLUSTER INDEX:", cluster_idx)

        # features = {}
        transform_feat = torch.nn.Linear(1000, 2048).to(local_rank)
        def hook(layer, inp, out):
            features['feat'] = inp[0]
            # feature_fc = layer(inp[0])
            # with torch.no_grad():
                # out_fc = torch.nn.functional.linear(inp[0], layer.weight, layer.bias)
            dist.all_reduce(features['feat'], op=torch.distributed.ReduceOp.SUM)
            # return out_fc
        # net.fc.register_forward_hook(hook)
        # if dist.get_rank() == 0:
        def backward_hook(module, grad_input, grad_output):
            # print("Backward Hook executed")
    # 检查梯度是否为 None，如果是，则处理逻辑
            for param in module.parameters():
                if param.grad is None:
                    # print(f"Gradient not received for {param.name}, setting it to zero.")
                    param.grad = torch.zeros_like(param)
        # net.module.fc.register_forward_hook(hook)
        # net.module.fc.register_full_backward_hook(backward_hook)
        # fc_reduction = torch.nn.Linear(64, 2048).to(local_rank)
        # def forward_hook_in_training_loop(model, image_adv, local_rank):
            # output = model(image_adv)
            # print('output after: ', output.shape)
            # output = torch.flatten(output, 1)
            # fc_features = model.module.fc(output)
            # print('output in forward shape:', fc_features.shape)
            # output = output.to(local_rank)
            # print(output.shape)
            # features['feat'] = fc_reduction(output)
            # fc_feature = model.module.fc(output)
            
            # dist.all_reduce(fc_features, op=torch.distributed.ReduceOp.SUM)
            # return fc_features

        for epoch in range(config['num_epoch']):
            g_net.train()
            # sampler.set_epoch(epoch)
            header = 'Class idx {}\tTrain Epoch {}:'.format(cluster_idx, epoch)

            # for images, _, _ in logger.log_every(train_loader, 50, header=header):
            for batch_idx, (images, _, _) in enumerate(train_loader):    
                images = images.to(local_rank)
                delta_im = g_net(noise).repeat(images.shape[0], 1, 1, 1)

                if config['norm'] == 'l2':
                    temp = torch.norm(delta_im.view(delta_im.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                    delta_im = delta_im * config['epsilon'] / temp
                else:
                    delta_im = torch.clamp(delta_im, -config['epsilon'] / 255., config['epsilon'] / 255)

                images_adv = torch.clamp(images + delta_im, 0, 1)
                target_labels = (torch.ones(len(images)).long() * cluster_idx + config['target_offset']) % num_clusters
                target_labels = target_labels.to(local_rank)
                # anchors = torch.index_select(cluster['centers'], dim=0, index=target_labels)
                anchors = torch.stack([cluster['centers'][i] for i in target_labels], dim=0).to(device)
                # print("Anchor Size: ", anchors.shape)
                output, fc_input  = net(normalize(images_adv))
                # fc_input = net.module.features['feat']
                # print("fc_input size :", fc_input.shape)
                
                # print("features['feat'] size: ", features['feat'].shape)
                # feat_transformed = transform_feat(fc_input)
                # print("Output size : ", output.shape)
                # print("Feat_transformed size: ", feat_transformed.shape)
                # print("anchors size: ", anchors.shape)
                # print('image_adv: ', normalize(images_adv).shape)
                # print('anchors: ', anchors.shape)
                # features['feat'] = forward_hook_in_training_loop(net, normalize(images_adv), local_rank)
                # print('feature:', features['feat'])
                loss = criterion(fc_input.log_softmax(dim=-1), anchors.softmax(dim=-1))
                # loss = criterion((0 * feat_transformed + features['feat']).log_softmax(dim=-1), anchors.softmax(dim=-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                if world_rank == 0:
                    print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")
                # logger.meters['train_loss'].update(loss.item(), n=len(images))

            with torch.no_grad():
                perturbation = g_net(noise)
            torch.save({'state_dict': g_net.state_dict(), 'init_noise': noise, 'perturbation': perturbation}, os.path.join(config['output_dir'], f'perturbation_{cluster_idx}.pth'))
            # if world_rank == 0:
            # logger.clear()

def train_gnet_gray(args, config):
    world_size, world_rank, local_rank = initialize_world()
    # device = torch.device(f'cuda:{rank}')
    # hvd.init()
    # torch.cuda.set_device(hvd.local_rank())
    # print(torch.cuda.get_device_name(args.device))
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(32),
        transforms.RandomCrop(28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    # print(torch.cuda.get_device_name(args.device))
    normalize = normalize_list[config['normalize']]
    batch_size = config['batch_size']
    local_batch_size = batch_size // world_size + 1
    print("world_size: ", world_size, "BATCH", batch_size, "LOCAL_BATCHSIZE: ", local_batch_size)

    # net = get_surrogate(config['model'], config['num_classes']).eval().to(args.device)
    # sd = torch.load(config['checkpoint'], map_location='cpu')
    # net.load_state_dict(sd)

    # net = get_surrogate(config['model'], config['num_classes']).eval().to(device)
    # net, device = get_surrogate(config['model'], local_rank, config['num_classes'])
    # pretrained_model = models.resnet50(pretrained=True)
    # pretrained_dict = pretrained_model.state_dict()

    model = ResNetGrayWithFeature()

    # checkpoints = torch.load('checkpoints/RN50_imagenet.pth')
    # model.load_state_dict(torch.load('checkpoints/RN50_imagenet.pth'))
    # model_dict = model.state_dict()
    # checkpoints = {k: v for k, b in checkpoints.items() if k in model_dict and 'fc' not in k}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    # model_dict.update(checkpoints)
    # model.load_state_dict(model_dict)
    net, device = get_distributed_model(model, local_rank)
    # net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [rank], find_unused_parameters=True)
    # net,device = get_distributed_model(net)

    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    cluster = torch.load(config['cluster'], map_location='cpu')
    cluster['centers'] = cluster['centers'].to(local_rank)
    num_clusters = cluster['centers'].shape[0]

    train_dataset = DataFolderWithLabel(config['dataset']['config']['train'], cluster['pred_idx'], train_transform)
    # train_loader = DataLoader(train_dataset, batch_size=256, num_workers=8, pin_memory=True)
    # train_dataset = DataFolderWithLabel(config['dataset']['config']['train'], cluster['pred_idx'], train_transform)
    sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=world_rank)
    # train_loader = DataLoader(train_dataset, batch_size=local_batch_size, num_workers=4, pin_memory=True, sampler=sampler)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, sampler=sampler)
    # Using hvd
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replocas=hvd.size(), rank=hvd.rank())
    # train_loader = torch.utils.data.Dataloader(train_dataset, batch_size=256, num_workers=8, pin_memory=True, sampler=train_sampler)

    for cluster_idx in range(num_clusters):
        noise = torch.zeros((1, 1, 28, 28))  # 修改噪声形状
        noise.uniform_(0, 1)
        noise = noise.to(local_rank)
        if world_rank == 0 :
            print("Cluster Index:", cluster_idx)

        g_net = ResnetGenerator(1, 1, 64, norm_type='batch', act_type='relu')  # 修改生成网络通道数
        g_net, device = get_distributed_model(g_net, local_rank)
        # g_net.to(args.device)
        # g_net.cuda()

        optimizer = torch.optim.Adam(g_net.parameters(), lr=config['lr'], weight_decay=5e-4)
        # optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.name_parameters())
        # hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['num_epoch'] * len(train_loader),
                                                               eta_min=1e-6)
        criterion = torch.nn.KLDivLoss(reduction='batchmean')

        # logger = MetricLogger()

        # features = {}
        transform_feat = torch.nn.Linear(1000, 2048).to(local_rank)

        def hook(layer, inp, out):
            features['feat'] = inp[0]
            # feature_fc = layer(inp[0])
            # with torch.no_grad():
            # out_fc = torch.nn.functional.linear(inp[0], layer.weight, layer.bias)
            dist.all_reduce(features['feat'], op=torch.distributed.ReduceOp.SUM)
            # return out_fc

        # net.fc.register_forward_hook(hook)
        # if dist.get_rank() == 0:
        def backward_hook(module, grad_input, grad_output):
            # print("Backward Hook executed")
            # 检查梯度是否为 None，如果是，则处理逻辑
            for param in module.parameters():
                if param.grad is None:
                    # print(f"Gradient not received for {param.name}, setting it to zero.")
                    param.grad = torch.zeros_like(param)

        # net.module.fc.register_forward_hook(hook)
        # net.module.fc.register_full_backward_hook(backward_hook)
        # fc_reduction = torch.nn.Linear(64, 2048).to(local_rank)
        # def forward_hook_in_training_loop(model, image_adv, local_rank):
        # output = model(image_adv)
        # print('output after: ', output.shape)
        # output = torch.flatten(output, 1)
        # fc_features = model.module.fc(output)
        # print('output in forward shape:', fc_features.shape)
        # output = output.to(local_rank)
        # print(output.shape)
        # features['feat'] = fc_reduction(output)
        # fc_feature = model.module.fc(output)

        # dist.all_reduce(fc_features, op=torch.distributed.ReduceOp.SUM)
        # return fc_features

        for epoch in range(config['num_epoch']):
            g_net.train()
            # sampler.set_epoch(epoch)
            header = 'Class idx {}\tTrain Epoch {}:'.format(cluster_idx, epoch)

            # for images, _, _ in logger.log_every(train_loader, 50, header=header):
            for batch_idx, (images, _, _) in enumerate(train_loader):
                images = images.to(local_rank)
                delta_im = g_net(noise).repeat(images.shape[0], 1, 1, 1)

                if config['norm'] == 'l2':
                    temp = torch.norm(delta_im.view(delta_im.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                    delta_im = delta_im * config['epsilon'] / temp
                else:
                    delta_im = torch.clamp(delta_im, -config['epsilon'] / 255., config['epsilon'] / 255)

                images_adv = torch.clamp(images + delta_im, 0, 1)
                target_labels = (torch.ones(len(images)).long() * cluster_idx + config['target_offset']) % num_clusters
                target_labels = target_labels.to(local_rank)
                # anchors = torch.index_select(cluster['centers'], dim=0, index=target_labels)
                anchors = torch.stack([cluster['centers'][i] for i in target_labels], dim=0).to(device)
                # print("Anchor Size: ", anchors.shape)
                output, fc_input = net(normalize(images_adv))
                # fc_input = net.module.features['feat']
                # print("fc_input size :", fc_input.shape)

                # print("features['feat'] size: ", features['feat'].shape)
                # feat_transformed = transform_feat(fc_input)
                # print("Output size : ", output.shape)
                # print("Feat_transformed size: ", feat_transformed.shape)
                # print("anchors size: ", anchors.shape)
                # print('image_adv: ', normalize(images_adv).shape)
                # print('anchors: ', anchors.shape)
                # features['feat'] = forward_hook_in_training_loop(net, normalize(images_adv), local_rank)
                # print('feature:', features['feat'])
                loss = criterion(fc_input.log_softmax(dim=-1), anchors.softmax(dim=-1))
                # loss = criterion((0 * feat_transformed + features['feat']).log_softmax(dim=-1), anchors.softmax(dim=-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                if world_rank == 0:
                    print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")
                
                # logger.meters['train_loss'].update(loss.item(), n=len(images))

            with torch.no_grad():
                perturbation = g_net(noise)
            torch.save({'state_dict': g_net.state_dict(), 'init_noise': noise, 'perturbation': perturbation},
                       os.path.join(config['output_dir'], f'perturbation_{cluster_idx}.pth'))
            # if world_rank == 0:
            # logger.clear()


def train_gray(args, config):
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 将图像转换为单通道
        transforms.Resize(28),  # 调整图像大小为 28x28
        transforms.RandomCrop(28),  # 随机裁剪28x28
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转换为张量
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 将图像转换为单通道
        transforms.Resize(28),  # 调整图像大小为 28x28
        transforms.CenterCrop(28),  # 中心裁剪28x28
        transforms.ToTensor(),  # 转换为张量
    ])

    world_size, world_rank, local_rank = initialize_world()

    normalize = normalize_list[config['normalize']]

    num_classes = config['dataset']['config']['num_classes']

    train_dataset = DataFolderWithLabel(config['ae_dir'], None, transform=train_transform)
    sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=world_rank)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=4, pin_memory=True,
                              sampler=sampler)

    test_dataset = DataFolderWithLabel(config['dataset']['config']['test'], None, test_transform)
    sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=world_size, rank=world_rank)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=4, pin_memory=True,
                             sampler=sampler)

    net, device = get_target(config['model'], local_rank, num_classes)

    optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['num_epoch'] * len(train_loader),
                                                           eta_min=1e-6)

    criterion = torch.nn.CrossEntropyLoss()
    logger = MetricLogger()

    for epoch in range(config['num_epoch']):
        net.train()
        header = 'Train Epoch {}:'.format(epoch)

        for images, labels, _ in logger.log_every(train_loader, 50, header=header):
            images, labels = images.to(local_rank), labels.to(local_rank)

            logits = net(normalize(images))
            loss = criterion(logits, labels)

            pred_idx = torch.argmax(logits.detach(), 1)
            correct = (pred_idx == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            logger.meters['train_loss'].update(loss.item(), n=len(images))
            logger.meters['train_acc'].update(correct / len(images), n=len(images))

    net.eval()
    header = 'Test Epoch {}:'.format(epoch)
    for images, labels, _ in logger.log_every(test_loader, 50, header=header):
        images, labels = images.to(local_rank), labels.to(local_rank)

        with torch.no_grad():
            logits = net(normalize(images))
            loss = criterion(logits, labels)

        pred_idx = torch.argmax(logits.detach(), 1)
        correct = (pred_idx == labels).sum().item()

        logger.meters['test_loss'].update(loss.item(), n=len(images))
        logger.meters['test_acc'].update(correct / len(images), n=len(images))

    torch.save({'state_dict': net.state_dict()}, os.path.join(config['output_dir'], 'checkpoint.pth'))
    logger.clear()



def train(args, config):
    train_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    world_size, world_rank, local_rank = initialize_world()

    normalize = normalize_list[config['normalize']]

    num_classes = config['dataset']['config']['num_classes']

    train_dataset = DataFolderWithLabel(config['ae_dir'], None, transform=train_transform)
    sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=world_rank)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=4, pin_memory=True, sampler=sampler)

    test_dataset = DataFolderWithLabel(config['dataset']['config']['test'], None, test_transform)
    sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=world_size, rank=world_rank)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=4, pin_memory=True, sampler=sampler)

    net, device = get_target(config['model'], local_rank, num_classes)

    optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['num_epoch'] * len(train_loader), eta_min=1e-6)

    criterion = torch.nn.CrossEntropyLoss()
    logger = MetricLogger()

    for epoch in range(config['num_epoch']):
        net.train()
        header = 'Train Epoch {}:'.format(epoch)

        for images, labels, _ in logger.log_every(train_loader, 50, header=header):
            images, labels = images.to(local_rank), labels.to(local_rank)

            logits = net(normalize(images))
            loss = criterion(logits, labels)

            pred_idx = torch.argmax(logits.detach(), 1)
            correct = (pred_idx == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            logger.meters['train_loss'].update(loss.item(), n=len(images))
            logger.meters['train_acc'].update(correct / len(images), n=len(images))

    net.eval()
    header = 'Test Epoch {}:'.format(epoch)
    for images, labels, _ in logger.log_every(test_loader, 50, header=header):
        images, labels = images.to(local_rank), labels.to(local_rank)

        with torch.no_grad():
            logits = net(normalize(images))
            loss = criterion(logits, labels)

        pred_idx = torch.argmax(logits.detach(), 1)
        correct = (pred_idx == labels).sum().item()

        logger.meters['test_loss'].update(loss.item(), n=len(images))
        logger.meters['test_acc'].update(correct / len(images), n=len(images))

    torch.save({'state_dict': net.state_dict()}, os.path.join(config['output_dir'], 'checkpoint.pth'))
    logger.clear()


def generate(args, config):
    normalize = normalize_list[config['normalize']]
    num_classes = config['dataset']['config']['num_classes']

    cluster = torch.load(config['cluster'], map_location='cpu')
    num_clusters = cluster['centers'].shape[0]

    noise = []
    for i in range(num_clusters):
        noise.append(torch.load(os.path.join(config['perturbation_dir'], f'perturbation_{i}.pth'), map_location='cpu')['perturbation'])
    noise = torch.cat(noise, dim=0)
    noise = torch.clamp(noise, -config['epsilon'] / 255., config['epsilon'] / 255)
    print(noise.shape)
    train_dataset = DataFolderWithClassNoise(config['dataset']['config']['train'], cluster['pred_idx'], noise=noise, resize_type=config['resize_type'])
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=8)

    count = [0 for _ in range(config['dataset']['config']['num_classes'])]
    output_dir = config['ae_dir']
    print(output_dir)
    for i in range(len(count)):
        Path(os.path.join(output_dir, str(i))).mkdir(parents=True, exist_ok=True)
    print('Done floder')

    logger = MetricLogger()
    header = 'Generate cluster-wise UEs:'

    count = [0 for _ in range(num_classes)]
    for i in range(len(count)):
        Path(os.path.join(config['output_dir'], '..', 'ae', str(i))).mkdir(parents=True, exist_ok=True)

    for images, ground_truth, _ in train_loader:
        images_adv = images

        ground_truth = ground_truth.tolist()

        for i in range(len(images)):
            gt = ground_truth[i]
            save_image(images_adv[i], os.path.join(config['output_dir'], '..', 'ae', str(gt), f'{count[gt]}.png'))
            count[gt] += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/stage_2.yaml')
    parser.add_argument('--experiment', '-e', type=str, default='uc_pets_cliprn50_rn18')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stage', type=int, default=2)
    args = parser.parse_args()
    yaml = YAML(typ='rt')
    with open(args.config, 'r') as f:
        config = yaml.load(f)[args.experiment]
    with open(config['data_config'], 'r') as f:
        data_config = yaml.load(f)[config['dataset']]
    
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    # config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)[args.experiment]
    # data_config = yaml.load(open(config['data_config'], 'r'), Loader=yaml.Loader)[config['dataset']]
    config['dataset'] = {'name': config['dataset'], 'config': data_config}
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)

    if args.stage == 1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        yaml.dump(config, open(os.path.join(config['output_dir'], '..', 'config.yaml'), 'w+'))
        train_gnet(args, config)

        #world_size = torch.cuda.device_count()
        #mp.spawn(train_gnet, args=(world_size, args, config), nprocs=world_size, join=True)
    elif args.stage == 2:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        yaml.dump(config, open(os.path.join(config['output_dir'], 'config.yaml'), 'w+'))
        generate(args, config)
        train(args, config)
    elif args.stage == 3:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        yaml.dump(config, open(os.path.join(config['output_dir'], '..', 'config.yaml'), 'w+'))
        train_gnet_gray(args, config)
    elif args.stage == 4:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        yaml.dump(config, open(os.path.join(config['output_dir'], 'config.yaml'), 'w+'))
        generate(args, config)
        train(args, config)
    else:
        raise KeyError
