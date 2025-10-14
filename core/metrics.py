import psutil
import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

def gpu_usage():
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024**2, info.total / 1024**2
    except Exception:
        return 0, 0

def ram_usage():
    mem = psutil.virtual_memory()
    return mem.used / 1024**2, mem.total / 1024**2

def timed_inference(fn, *args, **kwargs):
    start = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed
