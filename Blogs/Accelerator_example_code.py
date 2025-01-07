import argparse
import os
import evaluate
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,set_seed

)
from accelerate import (
    Accelerator,
    DataLoaderConfiguration,
    DistributedType,
)
MAX_GPU_BATCH_SIZE=16
EVAL_BATCH_SIZE=16

print(os.path.split(__file__)[-1].split(".")[0])
