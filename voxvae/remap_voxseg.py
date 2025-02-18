


def map_tensor(T, mapping):
    """
    Remaps the values 1, 2, 3 in tensor T to a, b, c respectively.
    T is assumed to be a numpy array containing values in {0, 1, 2, 3}.
    'mapping' is a dictionary (a,b,c)
    The value 0 remains 0.

    THANKS, ChatGPT!
    """
    a, b, c = mapping[0], mapping[1], mapping[2]

    # Compute the coefficients for f(x) = A*x**3 + B*x**2 + C*x
    A = (c + 3 * a - 3 * b) / 6
    B = (4 * b - 5 * a - c) / 2
    C = (18 * a - 9 * b + 2 * c) / 6

    # Apply the polynomial to every element of T.
    return A * T ** 3 + B * T ** 2 + C * T

if __name__ == "__main__":
    import numpy as np
    t = np.arange(4)
    print(map_tensor(t, (0.33, 0.66, 0.99)))