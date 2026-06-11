import matplotlib.pyplot as plt


def plot_positions(pos, clusters, step):

    plt.clf()

    plt.scatter(
        pos[:, 0],
        pos[:, 1],
        c=clusters,
        s=8,
        cmap="tab10",
        alpha=0.7
    )

    plt.title(f"Step {step}")
    plt.pause(0.01)