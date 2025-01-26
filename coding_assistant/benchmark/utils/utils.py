import math

import numpy as np
import torch
from torch.utils import benchmark

DEVICE = "cuda"


def efficiency(num_flops: int, exec_time: float):
    return (num_flops / exec_time / 10**12) if not math.isnan(exec_time) else math.nan


def benchmark_function(f, *inputs, repeats: int = 10, amp: bool = False, amp_dtype: torch.dtype = torch.float16,
                       **kw_inputs) -> tuple[benchmark.Timer, benchmark.Measurement]:
    def wrapper(*f_inputs, **f_kw_inputs):
        with torch.autocast(device_type=DEVICE, dtype=amp_dtype, enabled=amp):
            f(*f_inputs, **f_kw_inputs)

    timer = benchmark.Timer(
        stmt="fn_wrapper(*inputs, **kw_inputs)",
        globals={"fn_wrapper": wrapper, "inputs": inputs, "kw_inputs": kw_inputs},
        num_threads=torch.get_num_threads(),
    )

    measurement = timer.timeit(repeats)

    return timer, measurement


def time_function(func, *args, **kwargs) -> tuple[float, float]:
    _, measurement = benchmark_function(func, *args, **kwargs)

    times = np.array(measurement.times)
    return times.mean(), times.std()
