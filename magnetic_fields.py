import numpy as np
from scipy import special
from scipy.special import ellipk, ellipe


def coil_field(x, y, z, R):
    """
    Best version of the coil magnetic field so far, it uses cartesian coordinates instead of the classic cylindrical.
    Adapted from:
     "Simple Analytic Expressions for the Magnetic Field of a Circular Current Loop" Simpson, J et al, 2001, NASA.
    """
    rho = np.sqrt(z * z + y * y)
    r_sq = x * x + y * y + z * z
    alpha = np.sqrt(R * R + r_sq - 2. * R * rho)
    beta = np.sqrt(R * R + r_sq + 2. * R * rho)
    k = 1. - (alpha * alpha / (beta * beta))
    K = ellipk(k)  # Elliptic integral, first kind, as a function of k
    E = ellipe(k)
    C = 1.

    B_x = C / (2. * alpha * alpha * beta) * ((R * R - r_sq) * R + alpha * alpha * K)
    B_y = (C * y * x / (2. * alpha * alpha * beta * rho * rho)) * ((R * R + r_sq) * E - alpha * alpha * K)

    if B_x.ndim == 0:
        if y != 0:
            B_z = (z / y) * B_y
        elif z == 0:
            B_z = 0
        else:
            B_z = 0
            B_x = 0

    else:
        B_z = np.ones_like(B_y)
        B_z[y != 0] = (z / y) * B_y
        B_z[y == 0] = 0

        B_x[np.where((y == 0) & (z == 0))] = 0

        # B_x[rho > R] = 0
        # B_y[rho > R] = 0
        # B_z[rho > R] = 0

    return np.c_[B_x, B_y, B_z]


def coil_field_cylindrical(x, y, z, theta, R):
    """
    Magnetic field in a coil, cylindrical coordinate system. Adapted from different sources (list later);
    Formulas are correct, but the translation from cylindrical to cartesian has a mistake, so this field is currently
    not accurate. It may be worth debugging it, if comparing its results to the other field function is informative.
    """

    rho = np.sqrt(y * y + z * z)

    Bo = 1. / (2. * R)  # Central field = f(current, loop radius, perm. constant)
    alpha = rho / R  # Alpha = f(radius of measurement point, radius of loop)
    beta = x / R  # Beta = f(axial distance to meas. point, radius of loop)
    gamma = x / rho  # Gamma = f(axial distance, radius to meas. point)
    Q = (1 + alpha) ** 2 + beta ** 2  # Q = f(radius, distance to meas. point, loop radius)
    k = np.sqrt(4 * alpha / Q)  # k = f(radius, distance to meas. point, loop radius)
    K = ellipk(k * k)  # Elliptic integral, first kind, as a function of k
    E = ellipe(k * k)
    Bx = (Bo / (np.pi * np.sqrt(Q))) * (E * ((1.0 - alpha ** 2 - beta ** 2) / (Q - 4 * alpha)) + K)
    B_rho = (Bo * gamma / (np.pi * np.sqrt(Q))) * ((E * (1.0 + alpha ** 2 + beta ** 2) / (Q - 4 * alpha)) - K)

    if B_rho.ndim == 0:
        if np.isnan(B_rho):
            B_rho = 0
        if np.isinf(B_rho):
            B_rho = 0
        if np.isnan(Bx):
            Bx = 0
        if np.isinf(Bx):
            Bx = 0
    else:
        B_rho[np.isnan(B_rho)] = 0
        B_rho[np.isinf(B_rho)] = 0
        Bx[np.isnan(Bx)] = 0
        Bx[np.isinf(Bx)] = 0

    B = np.c_[Bx, np.sin(theta) * B_rho, np.cos(theta) * B_rho]

    return B


if __name__ == "__main__":
    cylindrical = coil_field_cylindrical(5, 5, 5, np.arctan(1), 10)
    cartesian = coil_field(5, 5, 5, 10)
    print(cylindrical)
    print(cartesian)
