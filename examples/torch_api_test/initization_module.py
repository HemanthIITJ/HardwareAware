import os
import torch.distributed as dist
import socket

def auto_init_dist():
    if dist.is_initialized():
        return

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')
    elif 'MASTER_ADDR' in os.environ and 'MASTER_PORT' in os.environ:
        try:
            dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')
        except ValueError:
            pass # Handle cases where rank/world_size might be missing
    elif dist.is_mpi_available():
        try:
            dist.init_process_group(backend='mpi')
        except RuntimeError:
            pass
    elif 'SLURM_PROCID' in os.environ and 'SLURM_NPROCS' in os.environ:
        try:
            dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo',
                                    init_method=f'tcp://{socket.gethostname()}:{os.environ.get("SLURM_TCP_PORT")}')
        except KeyError:
             pass # Handle cases where SLURM_TCP_PORT might be missing
    else:
        pass

auto_init_dist()