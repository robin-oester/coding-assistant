from dataclasses import dataclass
from deepspeed.profiling.flops_profiler import (get_model_profile, get_module_flops,
                                                get_module_macs, get_module_duration, params_to_string,
                                                macs_to_string, duration_to_string, flops_to_string)


@dataclass
class LayerStatistics:

    def __init__(self):
        self.stats = {}

        self.total_macs = 0
        self.total_duration = 0
        self.total_params = 0

    def register_layer(self, name):
        self.stats[name] = {
            "flops": 0,
            "macs": 0,
            "duration": 0,
            "params": 0,
        }

    def update_stats(self, name, layer):
        """
        Update statistics for a specific layer type.
        """
        assert name in self.stats, "Layer must be registered before updating the statistics"

        self.stats[name]["flops"] += get_module_flops(layer)
        self.stats[name]["macs"] += get_module_macs(layer)
        self.stats[name]["duration"] += get_module_duration(layer)
        self.stats[name]["params"] += getattr(layer, "__params__", 0)

    def set_total_statistics(self, total_macs, total_duration, total_params):
        self.total_macs = total_macs
        self.total_duration = total_duration
        self.total_params = total_params

    def get_stats(self):
        return self.stats

    def flops_repr(self, name, precision: int = 2) -> str:
        assert name in self.stats, "Layer must be registered before computing its representation"
        assert self.total_macs > 0 and self.total_duration > 0 and self.total_params > 0, "Set total first"

        flops = self.stats[name]["flops"]
        macs = self.stats[name]["macs"]
        duration = self.stats[name]["duration"]
        params = self.stats[name]["params"]

        items = [
            "{} = {:g}% Params".format(
                params_to_string(params),
                round(100 * params / self.total_params, precision)),
            "{} = {:g}% MACs".format(macs_to_string(macs),
                                     round(100 * macs / self.total_macs, precision)),
            "{} = {:g}% latency".format(
                duration_to_string(duration),
                round(100 * duration / self.total_duration, precision)),
            flops_to_string(round(flops / duration, precision) if duration else 0),
        ]
        return ", ".join(items)
