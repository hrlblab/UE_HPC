import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import os

import numpy as np
import os
import shutil
from pathlib import Path
from ruamel.yaml import YAML

from logger import MetricLogger
from utils import get_surrogate, normalize_list

from dataset.dataFolder import DataFolderWithLabel, DataFolderWithOneClass

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
import json
import torch.distributed as dist
import torch.multiprocessing as mp
import subprocess
import horovod.torch as hvd
import datetime


def initialize_world():
    num_gpus_per_node = torch.cuda.device_count()

    import subprocess
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

    get_master = "echo $(cat {} | sort | uniq | grep -v batch | grep -v login | head -1)".format(
        os.environ["LSB_DJOB_HOSTFILE"])
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


# def setup(rank, world_size):
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
    # try:
    #    result = subprocess.run("cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch", shell=True, capture_output=True, text=True, check=True)
    #    nodes = result.stdout.strip().split('\n')
    #    head = nodes[0] if nodes else None
    #    if head:
    #        os.environ['MASTER_ADDR'] = head
    #        print(f"Setting env_var MASTER_ADDR = {head}")
    #    else:
    #        print("No valid nodes found")

    # except subprocess.CalledProcessError as e:
    #    print(f"Error while getting nodes: {e}")
    ## os.environ['MASTER_ADDR'] = str(subprocess.check_output(get_master, shell=True))[2:-3]
    # os.environ['MASTER_PORT'] = '29500'
    # os.environ['WORLD_SIZE'] = os.environ.get('OMPI_COMM_WORLD_SIZE', '1')
    # os.environ['RANK'] = os.environ.get('OMPI_COMM_WORLD_RANK', '0')
    # os.environ['LOCAL_RANK'] = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')
    # dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)


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
            # device = get_device(device_name)
            device = get_device_from_name(device_name)
            model.to(device)
            model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            if sync_batch_norm:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            # device = get_device()
            device = get_device_from_name(device_name)
            model.to(local_rank)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], find_unused_parameters=True
            )
    return model, device_name


def cleanup():
    dist.destroy_process_group()



def dataset_median_embedding(args, config):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    normalize = normalize_list[args.normalize]

    num_classes = len(os.listdir(config[args.dataset]['test']['path']))
    net = get_surrogate(args.model, num_classes).eval().to(args.device)
    sd = torch.load(args.checkpoint, map_location='cpu')
    sd = sd.get('state_dict', sd)
    net.load_state_dict(sd)

    result = []

    for i in range(num_classes):
        train_dataset = DataFolderWithOneClass(config[args.dataset]['train']['path'], i, transform)
        train_loader = DataLoader(train_dataset, batch_size=64, num_workers=8)

        features = []
        def hook(layer, inp, out):
            features.append(inp[0].cpu())
        net.fc.register_forward_hook(hook)

        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            with torch.no_grad():
                net(normalize(images))

        features = torch.cat(features, dim=0)
        dis = squareform(pdist(features.numpy())).sum(axis=1)
        dis = torch.tensor(dis)
        idx = torch.argmin(dis)
        result.append(features[idx])

    torch.save(result, os.path.join(args.output_dir, f'{args.dataset}_{args.model}_median_embed.pth'))


def data_cluster(args, config):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    normalize = normalize_list[config['normalize']]

    train_dataset = DataFolderWithLabel(config['dataset']['config']['train'], transform)
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4)

    # sd = torch.load(config['checkpoint'], map_location='cpu')
    # sd = sd.get('state_dict', sd)
    net = get_surrogate(config['model'], config['num_classes']).eval().to(args.device)
    # net.load_state_dict(sd)

    features = []

    def hook(layer, inp, out):
        features.append(inp[0].cpu())

    net.fc.register_forward_hook(hook)

    for images, labels in train_loader:
        images, labels = images.to(args.device), labels.to(args.device)
        with torch.no_grad():
            net(normalize(images))

    features = torch.cat(features, dim=0)
    classifier = KMeans(n_clusters=config['num_clusters'])
    pred_idx = classifier.fit_predict(features.numpy())

    test_dataset = DataFolderWithLabel(config['dataset']['config']['test'], transform)
    test_loader = DataLoader(test_dataset, batch_size=256, num_workers=8)

    features = []
    for images, labels in test_loader:
        images, labels = images.to(args.device), labels.to(args.device)
        with torch.no_grad():
            net(normalize(images))

    features = torch.cat(features, dim=0)
    pred_idx_test = classifier.predict(features.numpy())

    result = {'pred_idx': torch.tensor(pred_idx), 'pred_idx_test': pred_idx_test, 'centers': torch.tensor(classifier.cluster_centers_)}
    num_clusters = result['centers'].shape[0]
    torch.save(result, os.path.join(config['output_dir'], f"{config['dataset']['name']}_{config['model'].lower()}_cluster{num_clusters}.pth"))


def datafolder_rename(args, config):
    train_dir = config['dataset']['config']['train']
    test_dir = config['dataset']['config']['test']

    class_list = sorted(os.listdir(train_dir))
    new_class = list(map(str, range(len(class_list))))

    name_mapping = {}
    name_mapping['map'] = {k: v for k, v in zip(class_list, new_class)}
    name_mapping['inv'] = {k: v for k, v in zip(new_class, class_list)}

    json.dump(name_mapping, open(os.path.join(train_dir, '..', 'name_mapping.json'), 'w+'))

    os.rename(train_dir, train_dir+'_raw')
    os.rename(test_dir, test_dir+'_raw')
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    for i in range(len(class_list)):
        shutil.copytree(os.path.join(train_dir+'_raw', class_list[i]), os.path.join(train_dir, new_class[i]))
        shutil.copytree(os.path.join(test_dir+'_raw', class_list[i]), os.path.join(test_dir, new_class[i]))

import argparse
if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--config', type=str)

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--function', '-f', type=str)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    yaml = YAML(typ='rt')
    with open(args.config, 'r') as f:
        config = yaml.load(f)[args.function]
    with open(config['data_config'], 'r') as f:
        data_config = yaml.load(f)[config['dataset']]
    # config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)[args.function]
    # data_config = yaml.load(open(config['data_config'], 'r'), Loader=yaml.Loader)[config['dataset']]
    config['dataset'] = {'name': config['dataset'], 'config': data_config}

    if args.function.startswith('median'):
        Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
        dataset_median_embedding(args, config)
    elif args.function.startswith('cluster'):
        Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
        data_cluster(args, config)
    elif args.function.startswith('rename'):
        datafolder_rename(args, config)
