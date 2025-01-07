import torch
import platform
import psutil
import multiprocessing
import subprocess

def get_hardware_info():
    info = {}
    info['system'] = platform.system()
    info['architecture'] = platform.machine()
    info['cpu'] = {
        'name': platform.processor(), 
        'cores': multiprocessing.cpu_count(),
        'architecture': platform.architecture()[0]
        }
    
    total_ram = psutil.virtual_memory().total
    info['memory'] = {
       'ram_total_gb': round(total_ram / (1024 ** 3), 2),
        'ram_usage_gb': round(psutil.virtual_memory().used / (1024 ** 3), 2),
        'rom_total_gb': None}
   
    try:
      if info['system']=='Linux':
        rom = subprocess.check_output(["df","-h","|", "grep", "/dev", "|", "awk", "{print $2}"], shell=True).decode().strip()
      if info['system']=='Darwin':
        rom = subprocess.check_output(["df","-h","|", "grep", "disk", "|", "awk", "{print $2}"], shell=True).decode().strip() 
      if rom:   info['memory']['rom_total_gb'] = float(rom[:-1]) if rom[-1]=='G' else  float(rom[:-1])/1024
    except Exception: pass
    
    if torch.cuda.is_available():
        info['cuda'] = {'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                        'capabilities': [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]}
    if torch.backends.mps.is_available():
        info['mps'] = {'available': torch.backends.mps.is_available(), 'built': torch.backends.mps.is_built() }
    info['backends'] = {'cpu': {
       'capability': torch.backends.cpu.get_cpu_capability()
       },
        'cudnn': {
           'is_available': torch.backends.cudnn.is_available(), 
            'version':torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            'enabled':torch.backends.cudnn.enabled , 
            'deterministic': torch.backends.cudnn.deterministic ,
            'benchmark': torch.backends.cudnn.benchmark , 
            'benchmark_limit':torch.backends.cudnn.benchmark_limit
                                 },
                        'cusparselt': {'is_available': torch.backends.cusparselt.is_available() , 'version': torch.backends.cusparselt.version() if torch.backends.cusparselt.is_available()else None },
                        'mha' : { 'fastpath_enabled':torch.backends.mha.get_fastpath_enabled() },
                        'mkl' : {'is_available':torch.backends.mkl.is_available() },
                        'mkldnn' : {'is_available':torch.backends.mkldnn.is_available() },
                        'nnpack' : {'is_available':torch.backends.nnpack.is_available() },
                        'openmp': {'is_available': torch.backends.openmp.is_available() },
                        # 'opt_einsum': {'is_available': torch.backends.opt_einsum.is_available()}
                        }
    return info

# if __name__ == '__main__':
#     hardware_info = get_hardware_info()
#     import json
#     print(json.dumps(hardware_info, indent=4))



def get_hardware_info():
    info = {}
    info['backends'] = [b for b in dir(torch.backends) if not b.startswith('_')]
    info['cpu'] = {
        'physical_cores': psutil.cpu_count(logical=False),
        'logical_cores': psutil.cpu_count(logical=True),
        'architecture': platform.machine()
    }
    vm = psutil.virtual_memory()
    info['ram'] = {
        'total': vm.total,
        'available': vm.available,
        'used': vm.used,
        'percent': vm.percent
    }
    du = psutil.disk_usage('/')
    info['rom'] = {
        'total': du.total,
        'used': du.used,
        'free': du.free,
        'percent': du.percent
    }
    if torch.cuda.is_available():
        info['cuda'] = {
            'device_count': torch.cuda.device_count(),
            'devices': [torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())]
        }
    return info

# if __name__ == "__main__":
#     hardware_details = get_hardware_info()
#     import json
#     print(json.dumps(hardware_details, indent=4))