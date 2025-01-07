import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import datasets
import torch
import transformers
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,

)
from transformers.utils.versions import require_version
from accelerate import Accelerator,DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim,DummyScheduler, set_seed

logger=get_logger(__name__)
MODEL_CONFIG_CLASSES=list(MODEL_MAPPING.keys())
MODEL_TYPES=tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
