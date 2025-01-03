import torch
import torch.backends

class CodeSignature_AZ_v1_Unique:
    """
    Automatically inspects and details PyTorch backend configurations.
    """
    _signature_ = "AZ_v1"  # Unique signature

    @classmethod
    def fetch_backend_details_unique(cls):
        """
        Analyzes and returns comprehensive details of PyTorch backends.
        """
        details = {"signature": cls._signature_}
        for name in dir(torch.backends):
            if not name.startswith("__"):
                try:
                    backend = getattr(torch.backends, name)
                    if hasattr(backend, '__dict__'):
                        module_info = {}
                        for key in dir(backend):
                            if not key.startswith("__"):
                                try:
                                    attr = getattr(backend, key)
                                    if callable(attr):
                                        try:
                                            module_info[key] = attr()
                                        except Exception as e:
                                            module_info[f"{key} (callable)"] = f"Error calling: {e}"
                                    else:
                                        module_info[key] = attr
                                except Exception as e:
                                    module_info[key] = f"Error accessing: {e}"
                        details[name] = module_info
                    else:
                        details[name] = str(backend)
                except Exception as e:
                    details[name] = f"Error inspecting: {e}"
        return details
import torch
import torch.backends
from typing import Optional

class CodeSignature_AX_v1_Unique:
    """
    Automatically detects and details the PyTorch CUDA backend configuration.
    """
    _signature_ = "AX_v1"

    @classmethod
    def get_cuda_backend_details_auto(cls):
        """
        Fetches and returns detailed information about the PyTorch CUDA backend.
        """
        cuda_details = {"signature": cls._signature_}

        if hasattr(torch.backends, 'cuda'):
            cuda_backend = torch.backends.cuda
            cuda_details['is_built'] = cuda_backend.is_built()

            cuda_details['matmul'] = {}
            if hasattr(cuda_backend, 'matmul'):
                cuda_details['matmul']['allow_tf32'] = getattr(cuda_backend.matmul, 'allow_tf32', None)
                cuda_details['matmul']['allow_fp16_reduced_precision_reduction'] = getattr(cuda_backend.matmul, 'allow_fp16_reduced_precision_reduction', None)
                cuda_details['matmul']['allow_bf16_reduced_precision_reduction'] = getattr(cuda_backend.matmul, 'allow_bf16_reduced_precision_reduction', None)

            cuda_details['cufft_plan_cache'] = {}
            if hasattr(cuda_backend, 'cufft_plan_cache'):
                cuda_details['cufft_plan_cache']['size'] = getattr(cuda_backend.cufft_plan_cache, 'size', None)
                cuda_details['cufft_plan_cache']['max_size'] = getattr(cuda_backend.cufft_plan_cache, 'max_size', None)

            cuda_details['preferred_blas_library'] = cuda_backend.preferred_blas_library()
            cuda_details['preferred_linalg_library'] = cuda_backend.preferred_linalg_library()

            cuda_details['sdpa_params'] = "Inspect manually as instantiation requires tensors" # Not automatically instantiable without data
            cuda_details['flash_sdp_enabled'] = cuda_backend.flash_sdp_enabled()
            cuda_details['mem_efficient_sdp_enabled'] = cuda_backend.mem_efficient_sdp_enabled()
            cuda_details['math_sdp_enabled'] = cuda_backend.math_sdp_enabled()
            cuda_details['fp16_bf16_reduction_math_sdp_allowed'] = cuda_backend.fp16_bf16_reduction_math_sdp_allowed()
            cuda_details['cudnn_sdp_enabled'] = cuda_backend.cudnn_sdp_enabled()
            cuda_details['is_flash_attention_available'] = cuda_backend.is_flash_attention_available()

        return cuda_details

    @classmethod
    def get_cudnn_backend_details_auto(cls):
        """
        Fetches and returns detailed information about the PyTorch cuDNN backend.
        """
        cudnn_details = {"signature": cls._signature_ + "_cudnn"}
        if hasattr(torch.backends, 'cudnn'):
            cudnn_backend = torch.backends.cudnn
            cudnn_details['version'] = cudnn_backend.version()
            cudnn_details['is_available'] = cudnn_backend.is_available()
            cudnn_details['enabled'] = cudnn_backend.enabled
            cudnn_details['allow_tf32'] = cudnn_backend.allow_tf32
            cudnn_details['deterministic'] = cudnn_backend.deterministic
            cudnn_details['benchmark'] = cudnn_backend.benchmark
            cudnn_details['benchmark_limit'] = cudnn_backend.benchmark_limit
        return cudnn_details

# Example usage (no user interaction needed)
if __name__ == "__main__":
    cuda_info = CodeSignature_AX_v1_Unique.get_cuda_backend_details_auto()
    cudnn_info = CodeSignature_AX_v1_Unique.get_cudnn_backend_details_auto()
    import json
    print("CUDA Backend Details:")
    print(json.dumps(cuda_info, indent=4, default=str))
    print("\ncuDNN Backend Details:")
    print(json.dumps(cudnn_info, indent=4, default=str))