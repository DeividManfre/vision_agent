import psutil
import time
import torch
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
)


class SystemMetrics:
    def __init__(self):
        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(0)
            print("[INFO] NVML initialized for GPU metrics.")
        except Exception:
            self.handle = None
            print("[WARN] GPU metrics unavailable (NVML init failed).")

    def _get_vram(self) -> float:
        if not self.handle:
            return 0.0
        try:
            info = nvmlDeviceGetMemoryInfo(self.handle)
            return info.used / 1024**2
        except Exception:
            return 0.0

    def _get_ram(self) -> float:
        mem = psutil.virtual_memory()
        return mem.used / 1024**2

    def timed_inference(self, fn, *args, **kwargs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        ram_before = self._get_ram()
        vram_before = self._get_vram()

        start = time.perf_counter()
        result = fn(*args, **kwargs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()

        time.sleep(0.05)

        ram_after = self._get_ram()
        vram_after = self._get_vram()

        delta_ram = round(max(0.0, ram_after - ram_before), 2)
        delta_vram = round(max(0.0, vram_after - vram_before), 2)
        elapsed = round(end - start, 3)

        if delta_ram < 1.0:
            delta_ram = abs(delta_ram)
        if delta_vram < 1.0:
            delta_vram = abs(delta_vram)

        return result, elapsed, delta_ram, delta_vram
