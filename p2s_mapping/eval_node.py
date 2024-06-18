import os
import re
import time
import json
import tqdm
import yaml
import h5py
import torch

import pickle
import shutil
import logging
import argparse
import numpy as np
from easydict import EasyDict as edict
from collections import defaultdict

import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from utils.simple_tokenizer import tokenize
from PIL import Image
from torchvision import transforms

from models import build_model
from utils.utils import Timer, AverageMeter, all_gather


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

train_io_timer = Timer()
train_model_timer = Timer()
test_io_timer = Timer()
test_model_timer = Timer()

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',
                        default='_'.join(time.asctime(time.localtime(time.time())).split()),
                        type=str,
                        help='Define exp name')
    parser.add_argument('--config_path',
                        required=True,
                        type=str,
                        help='Select config file')
    parser.add_argument('--local_rank',
                        type=int,
                        default=0)
    parser.add_argument('--ckpt_path',
                        required=True,
                        type=str,
                        help='Select checkpoint file')
    args = parser.parse_args()
    return args


def get_logger(cur_path):
    rank = dist.get_rank()
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    logger.propagate = False

    # create console handlers for master process
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(cur_path, f'log_rank{rank}.txt'), 'w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    writer = None

    return logger, writer


def Node2Text_sample(Node2Text, random=False):
    Node2Text_result = []
    for i in range(len(Node2Text)):
        if random:
            sel = np.random.choice(Node2Text[i], size=min(
                25, len(Node2Text[i])), replace=False)
            text = " ".join(sel)
            text = " ".join(text.split()[:25])
        else:
            # text = " ".join(Node2Text[i])
            text = ";".join(Node2Text[i])
        Node2Text_result.append(text)
    return Node2Text_result


def main(args, config, rank, world_size):


    ##########  MODEL
    net = build_model(config.MODEL, ckpt_path=args.ckpt_path).cuda()
    # logger.info(net)

    ##########  OPTIMIZER
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], 
                                                    broadcast_buffers=False, find_unused_parameters=False)


    ## prepare data
    # VerbNet
    n_px = 224
    transform = transforms.Compose([
                transforms.Resize(
                    n_px, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(n_px),
                lambda x: x.convert('RGB'),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
    
    data_dir = "../Data"
    mapping_node_index = pickle.load(
        open(f"{data_dir}/mapping_node_index.pkl", "rb"))
    verbnet_topology = pickle.load(
        open(f"{data_dir}/verbnet_topology_898.pkl", "rb"))
    Father2Son, objects = verbnet_topology["Father2Son"], verbnet_topology["objects"]
    objects = np.array(objects)
    node2id = {objects[i]: i for i in range(len(objects))}
    Node2Text = json.load(open(f"{data_dir}/Node2Description_898.json", "r"))
    Node2Text = [Node2Text[objects[i]]
                        for i in mapping_node_index]
    text_node = Node2Text_sample(Node2Text, random=False)
    text_node = tokenize(
        text_node, context_length=config.MODEL.CLIP.text.context_length)
    orig_image = Image.open("../asset/HAA500_kayaking_001_00009.jpg") # HAA500_pottery_wheel_004_00387
    image = transform(orig_image).numpy()
    batch = {
        "image": torch.from_numpy(image).unsqueeze(0).cuda(),
        "text_node": torch.from_numpy(np.array(text_node)).cuda()
    }
    output = net(batch)
    pred = output["p"].squeeze().detach().cpu().numpy()
    for idx in np.argsort(-pred)[:20]:
        print(pred[idx], objects[mapping_node_index[idx]])


if __name__ == '__main__':
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print("RANK %d WORLD_SIZE %d " % (rank, world_size))
    else:
        rank = -1
        world_size = -1

    args = parse_arg()
    loader = yaml.FullLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    config = edict(yaml.load(open(args.config_path, 'r'), Loader=loader))

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    config_ = ckpt["config"]
    config_.TEST = config.TEST
    config_.MODEL.CLIP.initial.weight = None
    config_.MODEL.CLIP.with_Lo = False
    config = config_
    
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])

    seed = config.SEED if 'SEED' in config else 0
    seed = seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    main(args, config, rank, world_size)
