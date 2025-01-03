import torch

class ZeusBackendDetective_ZBD:
    """
    🕵️‍♂️ Automatically detects and provides details about the user's CUDA backend.

    This class leverages PyTorch's CUDA backend introspection capabilities to
    automatically gather information about the user's CUDA setup without
    any required user input. It focuses on providing a comprehensive overview
    of relevant backend configurations.
    """
    _signature_zbd = "Crafted with precision by ZeusBackendDetective_ZBD"

    def __init__(self):
        self.backend_details = self._analyze_backend()

    def _analyze_backend(self):
        details = {}
        details['cuda_available'] = torch.cuda.is_available()
        if details['cuda_available']:
            details['cuda_built'] = torch.backends.cuda.is_built()
            details['device_count'] = torch.cuda.device_count()
            details['current_device'] = torch.cuda.current_device()
            details['device_name'] = torch.cuda.get_device_name(details['current_device'])

            # cuFFT
            details['cufft'] = {
                'plan_cache_max_size': torch.backends.cuda.cufft_plan_cache.max_size,
                'plan_cache_size': torch.backends.cuda.cufft_plan_cache.size
            }

            # cuBLAS
            details['cublas'] = {
                'allow_tf32': torch.backends.cuda.preferred_blas_library() == torch._C._BlasBackend.Cublas,
                'allow_tf32_settable': hasattr(torch.backends.cuda.cuBLASModule, 'allow_tf32'),
                'allow_fp16_reduced_precision_reduction': getattr(torch.backends.cuda.cuBLASModule, 'allow_fp16_reduced_precision_reduction', None),
                'allow_bf16_reduced_precision_reduction': getattr(torch.backends.cuda.cuBLASModule, 'allow_bf16_reduced_precision_reduction', None),
            }

            # Linalg
            details['linalg'] = {
                'preferred_library': str(torch.backends.cuda.preferred_linalg_library()).split('.')[1]
            }

            # SDP
            details['sdp'] = {
                'flash_enabled': torch.backends.cuda.flash_sdp_enabled(),
                'mem_efficient_enabled': torch.backends.cuda.mem_efficient_sdp_enabled(),
                'math_enabled': torch.backends.cuda.math_sdp_enabled(),
                'fp16_bf16_reduction_allowed': torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
            }
            details['flash_attention_available'] = torch.backends.cuda.is_flash_attention_available()
            # Note: Directly checking can_use_* attention methods requires SDPParams,
            # which necessitates tensor inputs, violating the "zero user involvement" principle.
            # We can only report on the general availability of flash attention.
            # For detailed attention mechanism checks, user interaction or assumptions about input tensors would be needed.
            # details['can_use_flash_attention'] = torch.backends.cuda.can_use_flash_attention(...)
            # details['can_use_efficient_attention'] = torch.backends.cuda.can_use_efficient_attention(...)
            # details['can_use_cudnn_attention'] = torch.backends.cuda.can_use_cudnn_attention(...)

        return details

    def get_backend_details(self):
        """Returns a dictionary containing detailed CUDA backend information."""
        return self.backend_details

    def __str__(self):
        """Provides a human-readable string representation of the backend details."""
        if not self.backend_details['cuda_available']:
            return "CUDA is not available."
        details_str = "CUDA Backend Details:\n"
        for key, value in self.backend_details.items():
            if isinstance(value, dict):
                details_str += f"  {key.upper()}:\n"
                for sub_key, sub_value in value.items():
                    details_str += f"    {sub_key}: {sub_value}\n"
            else:
                details_str += f"  {key.upper()}: {value}\n"
        details_str += f"\n{self._signature_zbd}"
        return details_str

# Example Usage:
if __name__ == "__main__":
    backend_info = ZeusBackendDetective_ZBD()
    print(backend_info)
    detailed_info = backend_info.get_backend_details()
    # Access specific details:
    if detailed_info['cuda_available']:
        print(f"\nNumber of CUDA devices: {detailed_info['device_count']}")
        if 'cufft' in detailed_info:
            print(f"cuFFT Plan Cache Max Size: {detailed_info['cufft']['plan_cache_max_size']}")