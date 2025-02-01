import matplotlib.pyplot as plt
import numpy as np

def plot_thigh_data(daphnetThigh, daphnetNames, i):
    """
    Plot thigh acceleration data for a specific dataset.
    Args:
        daphnetThigh (list): List of DataFrames containing thigh acceleration data.
        daphnetNames (list): List of dataset names.
        i (int): Index of the dataset to plot.
    """
    print(daphnetNames[i])
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(20, 16))
    fig.suptitle("Thigh Data from " + daphnetNames[i])
    plt.xlabel("Time")

    df = daphnetThigh[i]
    df = df[df.annotations > 0]  # Filter out rows with no annotations
    neg = df[df.annotations == 1]  # No freeze
    pos = df[df.annotations == 2]  # Freeze

    # Plot horizontal forward thigh acceleration
    ax1 = axes[0]
    ax1.plot(df.thigh_h_fd)
    ax1.set_ylabel("Horizontal Forward Thigh Acceleration")
    ax1.scatter(neg.index, neg.thigh_h_fd, c='orange', label="no freeze")
    ax1.scatter(pos.index, pos.thigh_h_fd, c='purple', label="freeze")
    ax1.legend()

    # Plot vertical thigh acceleration
    ax2 = axes[1]
    ax2.plot(df.thigh_v)
    ax2.set_ylabel("Vertical Thigh Acceleration")
    ax2.scatter(neg.index, neg.thigh_v, c='orange', label="no freeze")
    ax2.scatter(pos.index, pos.thigh_v, c='purple', label="freeze")
    ax2.legend()

    # Plot horizontal lateral thigh acceleration
    ax3 = axes[2]
    ax3.plot(df.thigh_h_l)
    ax3.set_ylabel("Horizontal Lateral Thigh Acceleration")
    ax3.scatter(neg.index, neg.thigh_h_l, c='orange', label="no freeze")
    ax3.scatter(pos.index, pos.thigh_h_l, c='purple', label="freeze")
    ax3.legend()

    # Plot overall thigh acceleration
    ax4 = axes[3]
    ax4.plot(df.thigh)
    ax4.set_ylabel("Overall Thigh Acceleration")
    ax4.scatter(neg.index, neg.thigh, c='orange', label="no freeze")
    ax4.scatter(pos.index, pos.thigh, c='purple', label="freeze")
    ax4.legend()

    plt.tight_layout()
    plt.show()


def plot_shank_data(daphnetShank, daphnetNames, i):
    """
    Plot shank acceleration data for a specific dataset.
    Args:
        daphnetShank (list): List of DataFrames containing shank acceleration data.
        daphnetNames (list): List of dataset names.
        i (int): Index of the dataset to plot.
    """
    print(daphnetNames[i])
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(20, 16))
    fig.suptitle("Shank Data from " + daphnetNames[i])
    plt.xlabel("Time")

    df = daphnetShank[i]
    df["shank"] = np.sqrt(df["shank_h_l"]**2 + df["shank_v"]**2 + df["shank_h_fd"]**2)
    df = df[df.annotations > 0]
    neg = df[df.annotations == 1]
    pos = df[df.annotations == 2]

    ax1 = axes[0]
    ax1.plot(df.shank_h_fd)
    ax1.set_ylabel("Horizontal Forward Shank Acceleration")
    ax1.scatter(neg.index, neg.shank_h_fd, c='orange', label="no freeze")
    ax1.scatter(pos.index, pos.shank_h_fd, c='purple', label="freeze")
    ax1.legend()

    ax2 = axes[1]
    ax2.plot(df.shank_v)
    ax2.set_ylabel("Vertical Shank Acceleration")
    ax2.scatter(neg.index, neg.shank_v, c='orange', label="no freeze")
    ax2.scatter(pos.index, pos.shank_v, c='purple', label="freeze")
    ax2.legend()

    ax3 = axes[2]
    ax3.plot(df.shank_h_l)
    ax3.set_ylabel("Horizontal Lateral Shank Acceleration")
    ax3.scatter(neg.index, neg.shank_h_l, c='orange', label="no freeze")
    ax3.scatter(pos.index, pos.shank_h_l, c='purple', label="freeze")
    ax3.legend()

    ax4 = axes[3]
    ax4.plot(df.shank)
    ax4.set_ylabel("Overall Shank Acceleration")
    ax4.scatter(neg.index, neg.shank, c='orange', label="no freeze")
    ax4.scatter(pos.index, pos.shank, c='purple', label="freeze")
    ax4.legend()

    plt.tight_layout()
    plt.show()


def plot_trunk_data(daphnetTrunk, daphnetNames, i):
    """
    Plot trunk acceleration data for a specific dataset.
    Args:
        daphnetTrunk (list): List of DataFrames containing trunk acceleration data.
        daphnetNames (list): List of dataset names.
        i (int): Index of the dataset to plot.
    """
    print(daphnetNames[i])
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(20, 16))
    fig.suptitle("Trunk Data from " + daphnetNames[i])
    plt.xlabel("Time")

    df = daphnetTrunk[i]
    df["trunk"] = np.sqrt(df["trunk_h_l"]**2 + df["trunk_v"]**2 + df["trunk_h_fd"]**2)
    df = df[df.annotations > 0]
    neg = df[df.annotations == 1]
    pos = df[df.annotations == 2]

    ax1 = axes[0]
    ax1.plot(df.trunk_h_fd)
    ax1.set_ylabel("Horizontal Forward Trunk Acceleration")
    ax1.scatter(neg.index, neg.trunk_h_fd, c='orange', label="no freeze")
    ax1.scatter(pos.index, pos.trunk_h_fd, c='purple', label="freeze")
    ax1.legend()

    ax2 = axes[1]
    ax2.plot(df.trunk_v)
    ax2.set_ylabel("Vertical Trunk Acceleration")
    ax2.scatter(neg.index, neg.trunk_v, c='orange', label="no freeze")
    ax2.scatter(pos.index, pos.trunk_v, c='purple', label="freeze")
    ax2.legend()

    ax3 = axes[2]
    ax3.plot(df.trunk_h_l)
    ax3.set_ylabel("Horizontal Lateral Trunk Acceleration")
    ax3.scatter(neg.index, neg.trunk_h_l, c='orange', label="no freeze")
    ax3.scatter(pos.index, pos.trunk_h_l, c='purple', label="freeze")
    ax3.legend()

    ax4 = axes[3]
    ax4.plot(df.trunk)
    ax4.set_ylabel("Overall Trunk Acceleration")
    ax4.scatter(neg.index, neg.trunk, c='orange', label="no freeze")
    ax4.scatter(pos.index, pos.trunk, c='purple', label="freeze")
    ax4.legend()

    plt.tight_layout()
    plt.show()


def plot_all_thigh_data(daphnetThigh, daphnetNames):
    """Plot thigh acceleration data for all datasets."""
    for i in range(len(daphnetThigh)):
        plot_thigh_data(daphnetThigh, daphnetNames, i)

def plot_all_shank_data(daphnetShank, daphnetNames):
    """Plot shank acceleration data for all datasets."""
    for i in range(len(daphnetShank)):
        plot_shank_data(daphnetShank, daphnetNames, i)

def plot_all_trunk_data(daphnetTrunk, daphnetNames):
    """Plot trunk acceleration data for all datasets."""
    for i in range(len(daphnetTrunk)):
        plot_trunk_data(daphnetTrunk, daphnetNames, i)


def plot_all_data(daphnetThigh, daphnetShank, daphnetTrunk, daphnetNames, i):
    """
    Plot thigh, shank, and trunk acceleration data for a specific dataset.
    Args:
        daphnetThigh (list): List of DataFrames containing thigh acceleration data.
        daphnetShank (list): List of DataFrames containing shank acceleration data.
        daphnetTrunk (list): List of DataFrames containing trunk acceleration data.
        daphnetNames (list): List of dataset names.
        i (int): Index of the dataset to plot.
    """
    plot_thigh_data(daphnetThigh, daphnetNames, i)
    plot_shank_data(daphnetShank, daphnetNames, i)
    plot_trunk_data(daphnetTrunk, daphnetNames, i)

def plot_all_datasets(daphnetThigh, daphnetShank, daphnetTrunk, daphnetNames):
    """Plot thigh, shank, and trunk acceleration data for all datasets."""
    for i in range(len(daphnetThigh)):
        plot_all_data(daphnetThigh, daphnetShank, daphnetTrunk, daphnetNames, i)