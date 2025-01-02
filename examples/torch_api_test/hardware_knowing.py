
import platform
import multiprocessing
import os
import subprocess

def detect_hardware1():
    hardware_info = {}
    hardware_info['os'] = platform.system()
    hardware_info['os_version'] = platform.version()
    hardware_info['processor'] = platform.processor()
    hardware_info['cpu_cores'] = multiprocessing.cpu_count()

    try:
        if hardware_info['os'] == 'Linux':
            with subprocess.Popen(['lscpu'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8') as process:
                stdout, _ = process.communicate()
                for line in stdout.splitlines():
                    if ":" in line:
                        key, value = map(str.strip, line.split(":", 1))
                        hardware_info[key.lower().replace(" ", "_")] = value
        elif hardware_info['os'] == 'Windows':
            import wmi
            c = wmi.WMI()
            for processor in c.Win32_Processor():
                hardware_info['cpu_name'] = processor.Name
                hardware_info['cpu_architecture'] = processor.Architecture
            for board in c.Win32_BaseBoard():
                hardware_info['motherboard_manufacturer'] = board.Manufacturer
                hardware_info['motherboard_product'] = board.Product
            for gpu in c.Win32_VideoController():
                hardware_info['gpu_name'] = gpu.Name
                hardware_info['gpu_adapter_ram'] = gpu.AdapterRAM
        elif hardware_info['os'] == 'Darwin':
            sysctl_commands = ['machdep.cpu.brand_string', 'hw.physicalcpu', 'hw.logicalcpu', 'hw.memsize']
            for cmd in sysctl_commands:
                try:
                    result = subprocess.run(['sysctl', cmd], capture_output=True, text=True, check=True)
                    key, value = map(str.strip, result.stdout.split(":", 1))
                    hardware_info[key.replace('.', '_')] = value
                except subprocess.CalledProcessError:
                    pass

    except ImportError:
        pass
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            hardware_info['cuda_available'] = True
            hardware_info['cuda_device_count'] = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            hardware_info['gpu_names'] = gpu_names
    except ImportError:
        pass
    except Exception:
        pass
    return hardware_info

# if __name__ == "__main__":
#     print(detect_hardware())

import platform
import os
import torch

def detect_hardware():
    hardware_info = {}
    hardware_info['os'] = platform.system()
    hardware_info['os_version'] = platform.version()
    hardware_info['processor'] = platform.processor()
    hardware_info['cpu_cores'] = os.cpu_count()
    hardware_info['has_cuda'] = torch.cuda.is_available()
    if hardware_info['has_cuda']:
        hardware_info['cuda_devices'] = torch.cuda.device_count()
        gpu_info = []
        for i in range(hardware_info['cuda_devices']):
            gpu_info.append(torch.cuda.get_device_name(i))
        hardware_info['cuda_names'] = gpu_info
    return hardware_info

if __name__ == "__main__":
    print(detect_hardware())
