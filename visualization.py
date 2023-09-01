import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "u_momentum",
    "v_momentum",
    "get_rhs",
    "get_coeff_mat_modified",
    "pres_correct",
    "update_velocity",
    "check_divergence_free"
]


def check_divergence_free(imax, jmax, dx, dy, u, v):
    plt.scatter(np.arange(iteration)[0:-1], res_history, s=20, alpha=0.5)
    plt.xlabel("Iterations")
    plt.ylabel("Residual")
    plt.title("Convergence History")
    ax = plt.gca()
    ax.set_yscale("log")




