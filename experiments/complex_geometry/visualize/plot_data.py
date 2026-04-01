import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.interpolate import griddata


def load_cfd_txt(path):
    """
    Expected format:
    cellnumber, x-coordinate, y-coordinate, pressure, x-velocity, y-velocity
    Example of row:
    1, 1.974143386E-01, 2.257942967E-02, 1.867345766E-05, 1.086770506E-03,-2.344900813E-03
    """
    df = pd.read_csv(path, header=None, sep=",", skipinitialspace=True, skiprows=1,
                     names=["cell", "x", "y", "p", "vx", "vy"], dtype=float)
    return df

def plot_dataset_all_points(df, title=None, draw_quiver=True):
    x = df["x"].values
    y = df["y"].values
    p = df["p"].values
    vx = df["vx"].values
    vy = df["vy"].values
    speed = np.sqrt(vx**2 + vy**2)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    sc0 = axs[0].scatter(x, y, c=p, s=8, cmap="viridis", marker='o')
    axs[0].set_title((title or "") + " — pressure (points)")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    cbar0 = fig.colorbar(sc0, ax=axs[0])
    cbar0.set_label("pressure")

    sc1 = axs[1].scatter(x, y, c=speed, s=8, cmap="inferno", marker='o')
    axs[1].set_title((title or "") + " — speed (|v|) (points)")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    cbar1 = fig.colorbar(sc1, ax=axs[1])
    cbar1.set_label("speed")

    if draw_quiver:
        axs[2].quiver(x, y, vx, vy, speed, angles='xy', scale_units='xy', scale=1)
        axs[2].set_title((title or "") + " — velocity vectors (every point)")
    else:
        axs[2].scatter(x, y, c='k', s=2)
        axs[2].set_title((title or "") + " — positions (quiver disabled)")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array(speed)
    fig.colorbar(sm, ax=axs[2]).set_label("speed (for color)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # df1 = load_cfd_txt("../../cylinder_viscid.csv")
    df2 = load_cfd_txt("../../../cylinder_inviscid.csv")

    # plot_dataset_all_points(df1, title="Dataset 1")

    plot_dataset_all_points(df2, title="Dataset 2")
