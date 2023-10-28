import numba
import scipy
import numpy as np


def _transform_reference(x, n):
    """Reference implementation."""
    X = np.zeros((len(x) - n + 1, n), dtype=x.dtype)
    for i in range(X.shape[0]):
        X[i, :] = x[i : i + n]
    return scipy.signal.hilbert(X, axis=1)


@numba.njit
def _transform_jit(H_x, h, x, n):
    """Just-in-time-compiled helper function."""
    for i in range(1, H_x.shape[0]):
        diff = x[n + i - 1] - x[i - 1]
        H_x[i, :] = np.roll(H_x[i - 1, :], -1)
        for d in range(n):
            H_x[i, d] += diff * h[d]


def transform(x, n):
    """
    Sliding window Hilbert transform.

    Parameters:
    -----------
    x : 1darray
        An array of time samples.
    n : int
        The window length.

    Returns:
    --------
    H : 2darray
        Every row of `H` contains the Hilbert transform of `n` consecutive samples of `x`.
    """
    H = np.zeros(n, dtype=complex)
    if n % 2 == 0:
        H[0] = 1
        H[n // 2] = 1
        H[1 : n // 2] = 2
    else:
        H[0] = 1
        H[1 : (n + 1) // 2] = 2

    h = scipy.fft.ifft(H)
    h = np.roll(scipy.fft.ifft(H), -1)

    # Hilbert transform of the first row
    H_x = np.empty((len(x) - n + 1, n), dtype=complex)
    H_x[0, :] = scipy.fft.fft(x[:n])
    H_x[0, :] = scipy.fft.ifft(H_x[0, :] * H)

    # the remaining rows
    _transform_jit(H_x, h, x, n)

    return H_x


if __name__ == "__main__":
    for n_samples, n_window in [
        (50, 40),
        (100, 30),
        (143, 142),
        (252, 251),
        (500, 490),
    ]:
        x_test = np.random.randn(n_samples)
        r1 = _transform_reference(x_test, n_window)
        r2 = transform(x_test, n_window)
        np.testing.assert_allclose(r1, r2)
    print("test passed.")
