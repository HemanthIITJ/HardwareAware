import torch
import platform
import psutil
import cpuinfo
import GPUtil


def get_hw_details():
    details = {}
    details['system'] = platform.system()
    details['nodes'] = 1
    details['cpu'] = {
        'name': cpuinfo.get_cpu_info()['brand_raw'],
        'cores': psutil.cpu_count(logical=False),
        'threads': psutil.cpu_count(logical=True),
        'arch': platform.machine(),
        'usage_percent': psutil.cpu_percent()
        }
    
    details['memory'] = {
        'total_gb': psutil.virtual_memory().total / (1024**3),
        'available_gb': psutil.virtual_memory().available / (1024**3),
        'used_gb': psutil.virtual_memory().used / (1024**3),
        'usage_percent': psutil.virtual_memory().percent
        }
    details['gpus'] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            details['gpus'].append(
                {
                'name': props.name,
                'multiprocessors': props.multi_processor_count,
                'arch': props.major,
                'total_memory_gb': props.total_memory / (1024**3),
                'memory_usage':torch.cuda.memory_allocated(i)/(1024**3),
                'memory_reserved':torch.cuda.memory_reserved(i)/(1024**3)
                }
    )
    elif GPUtil.getGPUs():
        for gpu in GPUtil.getGPUs():
            details['gpus'].append(
                {
                    'name': gpu.name,
                    'id': gpu.id,
                    'load': f'{gpu.load*100:.2f}%',
                    'free_memory_mb': gpu.memoryFree,
                    'used_memory_mb': gpu.memoryUsed,
                    'total_memory_mb': gpu.memoryTotal,
                    'temperature_c': gpu.temperature
                    }
        )
    details['disks'] = [{'device': p.device, 'mountpoint': p.mountpoint, 'usage': psutil.disk_usage(p.mountpoint).percent } for p in psutil.disk_partitions()]
    return details

if __name__ == "__main__":
    import json
    print(json.dumps(get_hw_details(), indent=4))