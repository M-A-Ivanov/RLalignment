import numpy as np


def base_vectors(n):
    """ Returns 3 orthogonal base vectors, the first one colinear to n.

        Parameters:
            n: The direction of the optical axis.
                ndarray, shape (3, )

        Returns:
            n, l, m: The three basis vectors
                ndarrays, shapes: (3, ), (3, ) and (3, )

    """
    # normalize n
    n = n / (np.sqrt(n.dot(n)))

    # choose two vectors perpendicular to n
    # choice is arbitrary since the coil is symetric about n
    if np.abs(n[0]) == 1:
        l = np.r_[n[2], 0, -n[0]]
    else:
        l = np.r_[0, n[2], -n[1]]

    l = l / (np.sqrt(l.dot(l)))
    m = np.cross(n, l)
    return n, l, m


def beam_creator(number_of_rays, cone_angle, n):
    """ Creates the initial fan of beams.
    Should add variable for (small and random) initial deflections in x and y
    to account for initial misalignment.
    Output:
        list_ic: list of lists, each containing 2 arrays 
        for position and for velocity.
            shape = [[beam1:array for X(1, 3), array for V(1, 3)],
                            [beam2:(1, 3), (1, 3)],
                                    .
                                    .
                                    .,
                            [beam_n:(1, 3), (1, 3)]   ].        
    """
    v0 = np.ones((1, number_of_rays, 3))
    x0 = np.zeros_like(v0)
    list_ic = []
    sigma = np.linspace(0, 2 * np.pi, number_of_rays + 1)
    sigma = sigma[:-1]
    theta = cone_angle * (np.pi / 180)
    v0[0, :, 1] = np.sin(theta) * np.sin(sigma)
    v0[0, :, 2] = np.sin(theta) * np.cos(sigma)

    R = get_rotational_matrix(n)
    for n in range(number_of_rays):
        v0[0, n, :] = R.dot(v0[0, n, :])
    v0 = v0 / np.sqrt(1 + np.cos(theta) ** 2)  # normalize
    for i in range(number_of_rays):
        list_ic.append((x0[0, i, :], v0[0, i, :]))
    return list_ic


def get_mag(x):
    """ Finds the magnitude of a vector x"""
    return np.sqrt(x.dot(x))


def points_in_cylinder(pt1, pt2, r, q):
    """Checks whether """
    vec = pt2 - pt1
    const = r * np.linalg.norm(vec)
    return np.where(np.dot(q - pt1, vec) >= 0 and np.dot(q - pt2, vec) <= 0
                    and np.linalg.norm(np.cross(q - pt1, vec)) <= const)


def get_rotational_matrix(n):
    # assert get_mag(n) == 1
    rot = np.arccos(n[0])
    return np.array(
        ([np.cos(rot), -np.sin(rot), 0], [np.sin(rot), np.cos(rot), 0], [0, 0, 1]))


def rotate_mgrid(n, X, Y, Z):
    assert get_mag(n) == 1
    rot = np.arccos(n[0])  # rotation to an angle rel. to the +x-axis
    X_rotated = np.cos(rot)*X + np.sin(rot)*Y
    Y_rotated = np.sin(rot)*X + np.cos(rot)*Y
    return X_rotated, Y_rotated, Z
