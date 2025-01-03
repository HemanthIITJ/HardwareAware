import torch
import torch.nn as nn
import torch.distributed as dist

class AdvParallelModelWrapper_AISHA(nn.Module):
    """
    A wrapper for PyTorch modules to enable automatic data parallelism (DataParallel or DistributedDataParallel).
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self._core_model = model
        self._parallel_model = None
        self._device = self._get_device()
        self._setup_parallel()

    def _get_device(self):
        """Automatically determine the device to use (CUDA if available, otherwise CPU)."""
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _setup_parallel(self):
        """Set up DataParallel or DistributedDataParallel if multiple GPUs are available."""
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                if dist.is_initialized():
                    local_rank = dist.get_rank()
                    self._parallel_model = nn.parallel.DistributedDataParallel(
                        self._core_model.to(self._device),
                        device_ids=[local_rank],
                        output_device=local_rank
                    )
                else:
                    self._parallel_model = nn.parallel.DataParallel(self._core_model).to(self._device)
            else:
                self._core_model.to(self._device)  # Move to single GPU
        else:
            self._core_model.to(self._device) # Move to CPU

    def forward_AISHA(self, input_ids, *args, **kwargs):
        """Forward pass through the model, automatically handling data parallelism."""
        input_ids = input_ids.to(self._device)
        if self._parallel_model:
            return self._parallel_model(input_ids, *args, **kwargs)
        return self._core_model(input_ids, *args, **kwargs)

# # Example Usage:
# if __name__ == '__main__':

#     # Sample Model
#     class SimpleModel(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.linear = nn.Linear(10, 2)

#         def forward(self, input_ids):
#             return self.linear(input_ids)

#     model = SimpleModel()
#     parallel_model = AdvParallelModelWrapper_AISHA(model)

#     # Sample Input
#     input_data = torch.randn(4, 10)  # Batch size 4, input size 10

#     # Perform forward pass
#     output = parallel_model.forward_AISHA(input_data)
#     print(output.shape)

import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.models as models

class AdvParallelModelWrapper_AISHA(nn.Module):
    """
    A robust, scalable, and generalized wrapper for PyTorch modules to enable automatic
    data parallelism (DataParallel or DistributedDataParallel), optimizing for large
    models and batch sizes.

    AISHA Signature: Always striving for efficiency and adaptability.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self._core_model = model
        self._parallel_model = None
        self._device = self._get_device()
        self._setup_parallel()

    def _get_device(self):
        """Automatically determine the appropriate device (CUDA if available, else CPU)."""
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _setup_parallel(self):
        """
        Set up data parallelism (either DataParallel or DistributedDataParallel) based on
        the availability of multiple GPUs and the distributed training environment.
        """
        self._core_model.to(self._device)  # Ensure the base model is on the primary device

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                if dist.is_initialized():
                    # Use DistributedDataParallel for multi-GPU distributed training (preferred for scalability)
                    local_rank = dist.get_rank()
                    self._parallel_model = nn.parallel.DistributedDataParallel(
                        self._core_model,
                        device_ids=[local_rank],
                        output_device=local_rank,
                        find_unused_parameters=True # Important for complex models
                    )
                else:
                    # Use DataParallel for single-node multi-GPU (simpler, but may have limitations with scaling)
                    self._parallel_model = nn.parallel.DataParallel(self._core_model)
            # If only one GPU, no need for wrapping, ensure model is on the GPU
        # If no CUDA, the base model is already on CPU

    def forward_AISHA(self, input_ids, *args, **kwargs):
        """
        Performs the forward pass, automatically handling data distribution if parallelization is active.
        Ensures all input tensors are moved to the correct device.

        Args:
            input_ids (torch.Tensor): The primary input tensor.
            *args:  Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The output of the model.
        """
        # Move primary input to the device
        input_ids = input_ids.to(self._device)

        # Move additional tensor arguments to the device
        args_to_device = tuple(arg.to(self._device) if isinstance(arg, torch.Tensor) else arg for arg in args)
        kwargs_to_device = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        if self._parallel_model:
            return self._parallel_model(*args_to_device, input_ids=input_ids, **kwargs_to_device)
        return self._core_model(input_ids,*args_to_device, **kwargs_to_device)

# Signature: Aisha performed this optimization.

# # Example Usage with torchvision model and dummy input
# if __name__ == '__main__':
#     # Initialize distributed environment if running in a distributed setting
#     # This is crucial for DistributedDataParallel
#     # if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#     #     dist.init_process_group(backend='nccl') # or 'gloo'

#     # Load a pre-trained ResNet model from torchvision
#     resnet_model = models.resnet18(pretrained=True)

#     # Wrap the model with the advanced parallel wrapper
#     parallel_resnet = AdvParallelModelWrapper_AISHA(resnet_model)

#     # Create a dummy input tensor (adjust shape based on the model's input requirements)
#     batch_size = 256
#     num_channels = 3
#     height = 224
#     width = 224
#     dummy_input = torch.randn(batch_size, num_channels, height, width)

#     # Perform the forward pass
#     output = parallel_resnet.forward_AISHA(dummy_input)

#     print(f"Output shape: {output.shape}")

#     # Clean up distributed environment if initialized
#     if dist.is_initialized():
#         dist.destroy_process_group()

import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.models as models
import os

class AdvParallelModelWrapper_AISHA(nn.Module):
    """
    A robust, scalable, and generalized wrapper for PyTorch modules to enable automatic
    data parallelism (DistributedDataParallel or DataParallel), optimized for large
    models and batch sizes.

    AISHA Signature: Adaptive Intelligence for System Harmony and Acceleration.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self._core_model = model
        self._parallel_model = None
        self._device = self._get_device()
        self._setup_parallel()

    def _get_device(self):
        """Automatically determine the most suitable device (CUDA if available, else CPU)."""
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _setup_parallel(self):
        """
        Sets up either DistributedDataParallel (preferred for scalability) or DataParallel
        based on the availability of multiple GPUs and the distributed training environment.
        """
        self._core_model.to(self._device)  # Move the base model to the determined device

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                if 'WORLD_SIZE' in os.environ and 'RANK' in os.environ:
                    # Assume DistributedDataParallel context if environment variables are set
                    if not dist.is_initialized():
                        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
                    local_rank = int(os.environ['LOCAL_RANK'])
                    self._parallel_model = nn.parallel.DistributedDataParallel(
                        self._core_model,
                        device_ids=[local_rank],
                        output_device=local_rank,
                        find_unused_parameters=True  # Handles models with conditional execution paths
                    )
                else:
                    # Fallback to DataParallel for single-machine multi-GPU
                    self._parallel_model = nn.parallel.DataParallel(self._core_model)
            # If only one GPU, no parallelization needed, model is already on GPU
        # If no CUDA, the base model remains on CPU

    def forward_AISHA(self, input_ids, *args, **kwargs):
        """
        Performs the forward pass through the model, automatically handling data distribution
        if parallelization is active. Ensures input tensors are moved to the correct device.

        Args:
            input_ids (torch.Tensor): The primary input tensor.
            *args:  Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The output of the model.
        """
        # Ensure primary input is on the correct device
        input_ids = input_ids.to(self._device)

        # Move additional tensor arguments to the device
        args_to_device = tuple(arg.to(self._device) if isinstance(arg, torch.Tensor) else arg for arg in args)
        kwargs_to_device = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        if self._parallel_model:
            return self._parallel_model(input_ids, *args_to_device, **kwargs_to_device)
        return self._core_model(input_ids, *args_to_device, **kwargs_to_device)

# Signature: Code crafted with precision by AISHA.

# Example Usage with torchvision model and dummy input
if __name__ == '__main__':
    # Load a pre-trained ResNet model from torchvision
    resnet_model = models.resnet50(pretrained=True)

    # Wrap the model with the advanced parallel wrapper
    parallel_resnet = AdvParallelModelWrapper_AISHA(resnet_model)

    # Define dummy input data with a potentially large batch size
    batch_size = 128
    num_channels = 3
    height = 256
    width = 256
    dummy_input = torch.randn(batch_size, num_channels, height, width)

    # Perform the forward pass
    output = parallel_resnet.forward_AISHA(dummy_input)

    print(f"Output shape: {output.shape}")

    # Clean up distributed environment if initialized (important for future runs)
    if dist.is_initialized():
        dist.destroy_process_group()
