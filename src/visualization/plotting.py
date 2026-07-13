from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def plot_positions(
    pos: np.ndarray, clusters: Optional[np.ndarray], labels: Optional[list[str]], step: int
) -> None:

    plt.clf()

    plt.scatter(pos[:, 0], pos[:, 1], c=clusters, s=8, cmap="tab10", alpha=0.7)

    handles: list[Line2D] = []

    cmap = plt.cm.get_cmap("tab10")

    for i, label in enumerate(labels):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cmap(i),
                markersize=8,
                label=label,
            )
        )

    plt.legend(handles=handles, title="Clusters")

    plt.title(f"Step {step}")
    plt.pause(0.01)
