import io
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm

# from the CPK colors in https://sciencenotes.org/wp-content/uploads/2019/07/CPK-JmolPeriodicTable.pdf
periodictable_colors = defaultdict(lambda: "pink")
periodictable_colors.update(
    {
        "H": "#FFFFFF",
        "He": "#D9FFFF",
        "Li": "#CC80FF",
        "Be": "#C2FF00",
        "B": "#FFB5B5",
        "C": "#909090",
        "N": "#3050F8",
        "O": "#FF0D0D",
        "F": "#90E050",
        "Ne": "#B3E3F5",
        "Na": "#AB5CF2",
        "Mg": "#8AFF00",
        "Al": "#BFA6A6",
        "Si": "#F0C8A0",
        "P": "#FF8000",
        "S": "#FFFF30",
        "Cl": "#1FF01F",
        "Ar": "#80D1E3",
    }
)


def draw_sphere(
    ax,
    x: float,
    y: float,
    z: float,
    size: float,
    color: str,
    alpha: float = 1.0,
    resolution: int = 25,
):
    """Draws a sphere surface centered in the coordinates (x,y,z) with size, color,
    transparency (alpha) and resoliution specified."""

    # 3d angles spanning the sphere
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    # origin-centered 3d surface of a sphere of radius <size>
    xs = size * np.outer(np.cos(u), np.sin(v))
    ys = size * np.outer(np.sin(u), np.sin(v))  # * 0.8
    zs = size * np.outer(np.ones(np.size(u)), np.cos(v))

    # translate the 3d sphere from origin to the new coordinates (0,0,0) -> (x,y,z)
    ax.plot_surface(
        x + xs,
        y + ys,
        z + zs,
        rstride=2,
        cstride=2,
        color=color,
        linewidth=0,
        alpha=alpha,
    )


def plot_cylinder(
    ax,
    x1: float,
    y1: float,
    z1: float,
    x2: float,
    y2: float,
    z2: float,
    radius: float,
    color: str = "gray",
    alpha: float = 1.0,
    resolution: int = 20,
):
    """Draws a cilinder surface between points (x1,y1,z1) and (x2,y2,z2), of the
    specified radius, color, transparency (alpha) and resolution.
    Note: Thanks to https://stackoverflow.com/questions/32317247/
    how-to-draw-a-cylinder-using-matplotlib-along-length-of-point-x1-y1-and-x2-y2"""

    # define origin
    origin = np.array([0, 0, 0])

    # axis and radius
    p0 = np.array([x1, y1, z1])
    p1 = np.array([x2, y2, z2])

    v = p1 - p0  # vector in direction of axis
    mag = norm(v)  # find magnitude of vector joining points
    v = v / mag  # unit vector in direction of axis

    # make some vector not in the same direction as v
    not_v = np.array([1, 0, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])

    n1 = np.cross(v, not_v)  # make vector perpendicular to v
    n1 /= norm(n1)  # normalize n1

    n2 = np.cross(v, n1)  # make unit vector perpendicular to v and n1

    # surface ranges over t from 0 to length of axis and 0 to 2*pi
    t = np.linspace(0, mag, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)

    # use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)

    # generate coordinates for surface
    X, Y, Z = [
        p0[i]
        + v[i] * t
        + radius * np.sin(theta) * n1[i]
        + radius * np.cos(theta) * n2[i]
        for i in [0, 1, 2]
    ]
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha)


def create_molecule_object(atom_symbols: list, positions: list, adjacency_matrix: list):
    """
    return a dictionary with information about the molecule
    """

    molecule_data = {}

    atom_data = []
    for i in range(len(atom_symbols)):
        x, y, z = positions[i, 0], positions[i, 1], positions[i, 2]
        atom_symbol = atom_symbols[i]
        atom_data.append(
            {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "atom_symbol": atom_symbol,
                "atom_index": i,
            }
        )

    bond_data = []
    for i in range(len(atom_symbols)):
        for j in range(i + 1, len(atom_symbols)):
            if adjacency_matrix[i, j] >= 1:  # not this wont draw the bonds
                bond_data.append(
                    {
                        "atom1": i,
                        "atom2": j,
                    }
                )

    molecule_data["atoms"] = atom_data
    molecule_data["bonds"] = bond_data

    return molecule_data


def plot_3dmolecule(
    molecule_info: dict,
    pred_pairwise_distances: torch.tensor,
    pred_adjacency_matrix: torch.tensor,
    iters: list,
    k: list,
    diff_losses: list,
    real_losses: list,
    filename: str = None,
    activation: list = None,
    azim: int = 55,
    elev: int = 55,
    bond_resolution: int = 15,
    atom_resolution: int = 20,
    dpi: int = 300,
    title: str = None,
):
    # create the figure
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(221, projection="3d")
    ax.set_aspect("equal")
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Reconstruction")

    # define the point of view (azimut and elevation)
    ax.view_init(elev=elev, azim=azim)

    # make planes transparent
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    plt.gca().axison = False

    # draw the bonds
    for bond in molecule_info["bonds"]:
        origin_found, destination_found = False, False
        x1, y1, z1 = None, None, None
        x2, y2, z2 = None, None, None

        origin_atom_idx = bond["atom1"]
        destination_atom_idx = bond["atom2"]

        for atom in molecule_info["atoms"]:
            if atom["atom_index"] == origin_atom_idx:
                x1, y1, z1 = atom["x"], atom["y"], atom["z"]
                origin_found = True

            if atom["atom_index"] == destination_atom_idx:
                x2, y2, z2 = atom["x"], atom["y"], atom["z"]
                destination_found = True

            if (origin_found) and (destination_found):
                break

        if (origin_found) and (destination_found):
            plot_cylinder(
                ax=ax,
                x1=x1,
                y1=y1,
                z1=z1,
                x2=x2,
                y2=y2,
                z2=z2,
                radius=0.1,
                color="lightgray",
                resolution=bond_resolution,
            )

    # draw the atoms
    for atom in molecule_info["atoms"]:
        draw_sphere(
            ax=ax,
            x=atom["x"],
            y=atom["y"],
            z=atom["z"],
            size=0.4,
            color=periodictable_colors[atom["atom_symbol"]],
            alpha=1.0,
            resolution=atom_resolution,
        )
        if (activation) and (atom["atom_index"] in activation):
            draw_sphere(
                ax=ax,
                x=atom["x"],
                y=atom["y"],
                z=atom["z"],
                size=0.6,
                color="red",
                alpha=0.2,
                resolution=atom_resolution,
            )

    # heatmap plot showing the atom connectivity (adjacency matrix)
    ax2 = fig.add_subplot(222)
    tensor_np = pred_adjacency_matrix.numpy()
    sns.heatmap(
        tensor_np, cmap="gray", cbar=False, linewidths=0.5, linecolor="black", ax=ax2
    )
    ax2.set_title("Predicted connectivity")
    ax2.set_xticks([])
    ax2.set_yticks([])

    # heatmap plot showing the pairwise-distances
    ax2 = fig.add_subplot(223)
    tensor_np = pred_pairwise_distances.numpy()
    sns.heatmap(
        tensor_np, cmap="magma", cbar=True, linewidths=0.5, linecolor="black", ax=ax2
    )
    ax2.set_title("Predicted pairwise-distances")
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax2 = fig.add_subplot(224)
    ax2.plot(iters, real_losses, label="Real Loss", color="black", linestyle="--")
    ax2.plot(iters, diff_losses, label="Diff Loss", color="black")

    # here add
    ax3 = ax2.twinx()
    ax3.plot(iters, k, label="K Value", color="red")
    ax3.set_ylabel("K Value")

    ax2.set_title("Loss evolution")
    ax2.set_xlabel("Number iterations")
    ax2.set_ylabel("MSE Loss")
    ax2.legend(loc="upper right")

    fig.suptitle(title, fontsize=14)
    if not filename:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        return buffer.getvalue()

    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
