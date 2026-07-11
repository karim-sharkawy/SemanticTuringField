import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_positions(pos, clusters, labels, step):

    plt.clf()

    plt.scatter(pos[:, 0], pos[:, 1], c=clusters, s=8, cmap="tab10", alpha=0.7)

    handles = []

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
