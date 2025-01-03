import torch
import torch.distributed as dist
import os
import socket
import subprocess
import logging
import re # Import the re module


class AdvancedAutoDistBackend:
    """
    An advanced class to automatically detect and configure the optimal
    distributed backend with optimized features and robust handling.
    """
    _SIGNATURE = "AdvancedAutoDistBackend_v2.1_2024"

    def __init__(self, logger_name="AutoDistBackend"):
       self.logger = self._setup_logger(logger_name)
       self.backend = self._detect_backend()
       self.world_size = self._get_world_size()
       self.rank = self._get_rank()
       self.init_method = self._get_init_method()
       self._configure_backend_env()
       self._init_process_group()
       self.logger.info(f"Distributed training initialized with: {self.get_info()}")


    def _setup_logger(self,name):
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            return logger

    def _detect_backend(self):
        if dist.is_nccl_available() and torch.cuda.is_available():
             self.logger.info("NCCL backend detected.")
             return "nccl"
        elif dist.is_gloo_available():
            self.logger.info("Gloo backend detected.")
            return "gloo"
        elif dist.is_mpi_available():
            self.logger.info("MPI backend detected.")
            return "mpi"
        else:
           self.logger.error("No suitable distributed backend found.")
           raise RuntimeError("No suitable distributed backend found.")

    def _get_world_size(self):
        if 'WORLD_SIZE' in os.environ:
             world_size = int(os.environ['WORLD_SIZE'])
             self.logger.info(f"World size from env: {world_size}")
             return world_size
        if dist.is_torchelastic_launched():
             world_size = int(os.environ['TORCHELASTIC_WORLD_SIZE'])
             self.logger.info(f"World size from torchelastic env: {world_size}")
             return world_size
        self.logger.info("World size set to 1 (single process).")
        return 1

    def _get_rank(self):
        if 'RANK' in os.environ:
            rank = int(os.environ['RANK'])
            self.logger.info(f"Rank from env: {rank}")
            return rank
        if dist.is_torchelastic_launched():
            rank = int(os.environ['TORCHELASTIC_LOCAL_RANK'])
            self.logger.info(f"Rank from torchelastic env: {rank}")
            return rank
        self.logger.info("Rank set to 0 (single process).")
        return 0

    def _get_init_method(self):
        if 'MASTER_ADDR' in os.environ and 'MASTER_PORT' in os.environ:
            init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
            self.logger.info(f"Init method using TCP: {init_method}")
            return init_method
        elif 'FILE_STORE_PATH' in os.environ:
            init_method = f"file://{os.environ['FILE_STORE_PATH']}"
            self.logger.info(f"Init method using file store: {init_method}")
            return init_method
        else:
           if self.world_size > 1 :
               self.logger.error("Environment needs MASTER_ADDR/PORT or FILE_STORE_PATH for distributed training.")
               raise ValueError("Environment needs MASTER_ADDR, MASTER_PORT or FILE_STORE_PATH for distributed training.")
           return None

    def _configure_backend_env(self):
        """Auto-configures backend env based on hardware and other parameters."""

        if self.backend == "nccl" and torch.cuda.is_available():
             self._configure_nccl_env()
        elif self.backend == "gloo":
              self._configure_gloo_env()

    def _configure_nccl_env(self):
         """Auto-configures NCCL environment variables."""
         try:
             if 'NCCL_SOCKET_IFNAME' not in os.environ:
                 iface = self._get_best_interface()
                 if iface:
                     os.environ["NCCL_SOCKET_IFNAME"] = iface
                     self.logger.info(f"NCCL_SOCKET_IFNAME set to: {iface}")
                 else:
                     self.logger.warning("No optimal network interface was found for NCCL, letting NCCL use default settings")
             if 'NCCL_DEBUG' not in os.environ:
                 os.environ['NCCL_DEBUG']  = "INFO"
                 self.logger.debug("NCCL_DEBUG set to INFO")

         except Exception as e:
             self.logger.error(f"Error configuring NCCL env variables: {e}")

    def _configure_gloo_env(self):
         try:
             if 'GLOO_SOCKET_IFNAME' not in os.environ:
                 iface = self._get_best_interface()
                 if iface:
                  os.environ["GLOO_SOCKET_IFNAME"] = iface
                  self.logger.info(f"GLOO_SOCKET_IFNAME set to : {iface}")
                 else:
                     self.logger.warning("No optimal network interface was found for GLOO, letting GLOO use default settings")
         except Exception as e:
            self.logger.error(f"Error configuring GLOO env variables: {e}")

    def _get_best_interface(self):
        """Attempts to auto-detect the best network interface."""

        try:
            ifaces = socket.if_nameindex()
            ipv4_addr = {}
            for iface in ifaces:
              name = iface[1]
              try:
                   ip_cmd = subprocess.run(["ip", "-4", "addr", "show", name], capture_output=True, text=True, check=True)
                   ip_output = ip_cmd.stdout
                   match = re.search(r'inet\s+([\d.]+)/',ip_output)
                   if match:
                        ipv4_addr[name] = match.group(1)
              except Exception as e:
                    self.logger.debug(f"Could not process interface {name} : {e}")

            if 'eth0' in ipv4_addr:
                return 'eth0'
            elif 'enp0s3' in ipv4_addr:
                return 'enp0s3'
            elif "lo" in ipv4_addr :
                 return None
            else:
                 return  list(ipv4_addr.keys())[0] if ipv4_addr else  None

        except Exception as e:
             self.logger.error(f"Error auto-detecting network interface: {e}")
             return None

    def _init_process_group(self):
        if self.world_size > 1 and self.init_method is not None:
            try:
                dist.init_process_group(backend=self.backend, init_method=self.init_method, rank=self.rank, world_size=self.world_size)
                self.logger.info("Process group initialized successfully.")
            except Exception as e:
                self.logger.error(f"Error during process group initialization: {e}")
                raise

    def get_info(self):
        return {
            "backend": self.backend,
            "world_size": self.world_size,
            "rank": self.rank,
            "init_method": self.init_method,
        }

    def cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()
            self.logger.info("Process group destroyed.")

    def __del__(self):
       self.cleanup()

if __name__ == '__main__':

   try:
        dist_config = AdvancedAutoDistBackend()
        print(f"Auto-Detected Distributed Configuration: {dist_config.get_info()}")

        if dist_config.world_size > 1 and dist.is_initialized():
           print(f"Rank {dist_config.rank}/{dist_config.world_size} process is active")
           tensor = torch.ones(1) * dist_config.rank
           dist.all_reduce(tensor)
           print(f"Rank {dist_config.rank} has tensor {tensor}")

   except Exception as e:
        print(f"An error has occurred: {e}")
   finally:
        print("Task has finished.")