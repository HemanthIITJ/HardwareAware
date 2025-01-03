import torch
import torch.nn as nn
import torch.distributed as dist
import os
from typing import Any, Tuple, Dict
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader


class ParallelModelWrapper:
    """
     A smart and robust wrapper for parallelizing PyTorch models using DataParallel or DistributedDataParallel.

     Handles various input types, model sizes, and batch sizes efficiently and gracefully.
     Provides robust error handling and logging.


    Signature:
        -  ParallelModelWrapper_v1.1.0_by_s1am04
    """

    def __init__(self, model: nn.Module, auto_parallelize: bool = True):
        self.model = model
        self.device = self._get_device()
        self.parallel_model = self._auto_parallelize() if auto_parallelize else model.to(self.device)
        self.auto_parallelize_flag = auto_parallelize

    def _get_device(self) -> torch.device:
        """
           Automatically detects and returns the appropriate device (CUDA if available, otherwise CPU).
        """
        if torch.cuda.is_available():

            return torch.device("cuda")

        else:
            return torch.device("cpu")

    def _is_distributed_env(self) -> bool:
        """
            Checks if the code is running in a distributed environment.
        """
        return dist.is_available() and dist.is_initialized()

    def _auto_parallelize(self) -> nn.Module:
        """
            Automatically selects and applies the best parallelization method (DDP or DP) based on the environment.
        """
        self.model.to(self.device)
        try:
            if self._is_distributed_env():
                local_rank = int(os.environ["LOCAL_RANK"])
                self.device = torch.device(f"cuda:{local_rank}")
                model = self.model.to(self.device)

                return nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                     find_unused_parameters=True # Ensure robustness

                )

            elif torch.cuda.device_count() > 1:
                device_ids = list(range(torch.cuda.device_count()))
                return nn.DataParallel(self.model, device_ids=device_ids,output_device=device_ids[0])

            return self.model
        except Exception as e:
             print(f"Auto-parallelization failed: {e}. Running model on single device.")
             return self.model

    def _prepare_input(self, input_data: Any) -> Any:
            """
                Moves input data to the correct device and handles different input types.
            """
            if isinstance(input_data, torch.Tensor):
                 input_data = input_data.to(self.device)
            elif isinstance(input_data, (list, tuple)):
                input_data = [self._prepare_input(item) for item in input_data]
            elif isinstance(input_data, dict):
                input_data = {k: self._prepare_input(v) for k, v in input_data.items()}

            return input_data

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
            """
                Forward pass through the parallelized model, handling multiple input types.
                Ensures all input tensors are moved to the correct device.
            """
            try:


                prepared_args = [self._prepare_input(arg) for arg in args]
                prepared_kwargs = {k: self._prepare_input(v) for k, v in kwargs.items()}


                if self.auto_parallelize_flag:

                    output = self.parallel_model(*prepared_args, **prepared_kwargs)
                else:
                    output = self.model(*prepared_args, **prepared_kwargs)

                return output

            except Exception as e:
                print(f"Error during forward pass: {e}")
                raise

    # Signature: ParallelModelWrapper_v1.1.0_by_s1am04



if __name__ == "__main__":
    # Example usage with torchvision models and dummy inputs

    # Define a dummy dataset
    class DummyDataset(Dataset):
        def __init__(self, size, input_shape):
            self.size = size
            self.input_shape = input_shape

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return torch.randn(self.input_shape)

    # Test for different input shape
    input_shapes = [(3,224,224),(3,128,128),(1,224,224)]

    for input_shape in input_shapes:
          # Create a ResNet-18 model
          model = models.resnet18(pretrained=False)  # Test with a non-pretrained model

          #Wrap the model
          parallel_model = ParallelModelWrapper(model)

          # Create the dummy dataset and dataloader
          dataset = DummyDataset(size=200, input_shape=input_shape)
          dataloader = DataLoader(dataset, batch_size=32)

          # Run the forward pass with the dummy data, including variable batch sizes
          for i, batch_input in enumerate(dataloader):
               try:
                    output = parallel_model(batch_input)
                    print(f"Batch {i},Input-shape: {input_shape}, Output shape: {output.shape}")

               except Exception as e:
                    print(f"Error in batch {i} for input_shape {input_shape}: {e}")

          # Example with more complex inputs: tuple of tensors, dictionary of tensors

          dummy_input1 = torch.randn(32, *input_shape)
          dummy_input2 = torch.randn(32, *input_shape)
          tuple_input = (dummy_input1, dummy_input2)

          dict_input = {"input1": dummy_input1, "input2": dummy_input2}

          try:
              tuple_output = parallel_model(*tuple_input)
              print(f"Tuple input, Input-shape: {input_shape} output shape: {tuple_output[0].shape}")

          except Exception as e:
              print(f"Error in tuple input, input_shape: {input_shape} with: {e}")

          try:
              dict_output = parallel_model(**dict_input)
              print(f"Dict input, Input-shape: {input_shape} output shape: {dict_output.shape}")

          except Exception as e:
               print(f"Error in dict input, input_shape: {input_shape} with {e}")

          # Test without automatic parallelization:
          parallel_model_no_auto = ParallelModelWrapper(model, auto_parallelize=False)
          try:

               output = parallel_model_no_auto(dummy_input1)
               print(f"No auto, Input-shape: {input_shape}, Output shape: {output.shape}")
          except Exception as e:
             print(f"Error in no_auto, input_shape: {input_shape} with {e}")

          print("------------------------------end----------------------------------------")