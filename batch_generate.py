#!/usr/bin/env python3
# coding=utf-8
import os
import logging
import time
import datetime
import torch
import copy
import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from megatron.data.data_utils import build_the_dataset
from transformers import AutoModelForCausalLM
import transformers.utils as transformer_utils
import multiprocessing as mp
from tqdm import trange

def generate_dataset(batch_size, start_seq_idx, end_seq_idx, mp_queue, prefetch_max=128):
    prefix = '/scratch/pile/standard/document.bin'
    if "deduped" in os.environ['MODEL']:
        prefix = 'orz/pile/deduped/document.bin'
