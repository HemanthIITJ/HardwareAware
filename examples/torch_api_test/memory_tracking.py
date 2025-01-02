# mypy: allow-untyped-defs
from re import A
import torchvision

import torch
from torch.distributed._tools import MemoryTracker


def run_one_model(net: torch.nn.Module, input: torch.Tensor):
    net.cuda()
    input = input.cuda()

    # Create the memory Tracker
    mem_tracker = MemoryTracker()
    # start_monitor before the training iteration starts
    mem_tracker.start_monitor(net)

    # run one training iteration
    net.zero_grad(True)
    loss = net(input)
    if isinstance(loss, dict):
        loss = loss["out"]
    loss.sum().backward()
    net.zero_grad(set_to_none=True)

    # stop monitoring after the training iteration ends
    mem_tracker.stop()
    # print the memory stats summary
    mem_tracker.summary()
    # plot the memory traces at operator level
    mem_tracker.show_traces()


# run_one_model(torchvision.models.resnet34(), torch.rand(32, 3, 224, 224, device="cuda"))

import torch
import torch.nn as nn
from torch.distributed._tools import MemoryTracker
import torchvision
import matplotlib.pyplot as plt
import os
from itertools import chain
import pickle
class AutoMemoryTracker(MemoryTracker):
    _DEFAULT_STATS_PATH = "memory_stats.pkl"
    _DEFAULT_PLOT_PREFIX = "memory_trace"

    def __init__(self, model: nn.Module, inputs: torch.Tensor, load_saved_stats: bool = False):
        super().__init__()
        self.model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.inputs = inputs.to(next(model.parameters()).device if list(model.parameters()) else "cpu")
        if load_saved_stats and os.path.exists(self._DEFAULT_STATS_PATH):
            self.load(self._DEFAULT_STATS_PATH)
            self._show_and_save_traces()
        else:
            self._track()
            self.save_stats(self._DEFAULT_STATS_PATH)
            self._show_and_save_traces()

    def _track(self):
        self.start_monitor(self.model)
        with torch.no_grad():
            _ = self.model(self.inputs)
        self.stop()
        self.summary()

    def _show_and_save_traces(self):
        self.show_traces()
        # Save all open figures
        figures = [plt.figure(n) for n in plt.get_fignums()]
        for i, fig in enumerate(figures):
            fig.savefig(f"{self._DEFAULT_PLOT_PREFIX}_{i}.png")
        plt.show()  # Keep plt.show() to also display the plots

    def show_traces(self, path: str = "") -> None:
        import matplotlib.pyplot as plt

        def _plot_figure(x, y_values, labels, fig_index):
            min_val = min(list(chain(*y_values))) * 0.999
            max_val = max(list(chain(*y_values))) * 1.001
            plt.figure(fig_index)
            for y, label in zip(y_values, labels):
                plt.plot(x, y, label=label)
            plt.xlabel("# Operator Calls")
            plt.ylabel("Memory (MB)")
            plt.legend()
            for marker_name, marker in self._markers.items():
                if marker_name == "fw_bw_boundary":
                    plt.plot(
                        [marker, marker],
                        [min_val, max_val],
                        "r",
                        lw=2,
                        label=marker_name,
                    )
                else:
                    plt.plot(
                        [marker, marker],
                        [min_val, max_val],
                        "k-",
                        lw=2,
                        label=marker_name,
                    )

        if path != "":
            self.load(path)

        y_1 = [gb for (name, gb) in self.memories_allocated.values()]
        y_2 = [gb for (name, gb) in self.memories_active.values()]
        y_3 = [gb for (name, gb) in self.memories_reserved.values()]
        x = list(range(len(y_1)))
        # Split figures when there is big difference between
        # "reserved_memory" and "allocated_memory" or "active_memory".
        fig_index = 0
        _plot_figure(
            x,
            [list(y_1), list(y_2), list(y_3)],
            ["allocated_memory", "active_memory", "reserved_memory"],
            fig_index
        )
        fig_index += 1
        _plot_figure(x, [list(y_1)], ["allocated_memory"], fig_index)
        fig_index += 1
        _plot_figure(x, [list(y_2)], ["active_memory"], fig_index)
        fig_index += 1
        _plot_figure(x, [list(y_3)], ["reserved_memory"], fig_index)

    def save_stats(self, path: str) -> None:
        """Save the stats using pickle during runtime if users want to plot the traces in other places like notebook."""
        stats = {
            "memories_allocated": self.memories_allocated,
            "memories_active": self.memories_active,
            "memories_reserved": self.memories_reserved,
            "markers": self._markers,
            "num_alloc_retries": self._num_cuda_retries,
        }

        with open(path, "wb") as f:
            pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        """Load the pickled memory stats to plot the traces or print the summary."""
        with open(path, "rb") as f:
            stats = pickle.load(f)

        self.memories_allocated = stats["memories_allocated"]
        self.memories_active = stats["memories_active"]
        self.memories_reserved = stats["memories_reserved"]
        self._markers = stats["markers"]
        self._num_cuda_retries = stats["num_alloc_retries"]

if __name__ == '__main__':
    mem=AutoMemoryTracker(torchvision.models.resnet34(), torch.rand(32, 3, 224, 224))
    # To load and plot saved stats:
    # mem_loaded = AutoMemoryTracker(torchvision.models.resnet34(), torch.rand(32, 3, 224, 224), load_saved_stats=True)