import numba
import scipy
import numpy as np


def _transform_reference(x, l_window):
    """Reference implementation."""
    X = np.zeros((len(x) - l_window + 1, l_window), dtype=x.dtype)
    for i in range(X.shape[0]):
        X[i, :] = x[i : i + l_window]
    return scipy.signal.hilbert(X, axis=1)


@numba.njit
def _transform_jit(H_x, h, x, l_window):
    """Just-in-time-compiled helper function."""
    for i in range(1, H_x.shape[0]):
        diff = x[l_window + i - 1] - x[i - 1]
        H_x[i, :] = np.roll(H_x[i - 1, :], -1)
        for d in range(l_window):
            H_x[i, d] += diff * h[d]


def transform(x, l_window):
    """
    Sliding window Hilbert transform.

    Parameters:
    -----------
    x : 1darray
        An array of time samples.
    l_window : int
        The window length.

    Returns:
    --------
    H : 2darray
        Every row of `H` contains the Hilbert transform of `l_window` consecutive samples of `x`.
    """
    W = np.zeros(l_window, dtype=complex)
    if l_window % 2 == 0:
        W[0] = 1
        W[l_window // 2] = 1
        W[1 : l_window // 2] = 2
    else:
        W[0] = 1
        W[1 : (l_window + 1) // 2] = 2

    w = scipy.fft.ifft(W)
    w = np.roll(scipy.fft.ifft(W), -1)

    # Hilbert transform of the first row
    H = np.empty((len(x) - l_window + 1, l_window), dtype=complex)
    H[0, :] = scipy.fft.fft(x[:l_window])
    H[0, :] = scipy.fft.ifft(H[0, :] * W)

    # the remaining rows
    _transform_jit(H, w, x, l_window)

    return H


def _sliding_transform_reference(x, l_window, n_lag):
    """Reference implementation."""
    H = _transform_reference(x, l_window)
    # extract the relevant samples
    h_x = np.zeros(x.shape, dtype=complex) * np.nan
    h_x[l_window - n_lag + 1 : 1 - n_lag] = H[1:, -n_lag]
    return h_x


@numba.njit
def _sliding_transform_jit(H, w, x, h_x, l_window, n_lag):
    """Just-in-time-compiled helper function."""
    for i in range(1, len(x) - l_window + 1):
        # update the Hilbert transform
        diff = x[l_window + i - 1] - x[i - 1]
        H = np.roll(H, -1)
        for d in range(l_window):
            H[d] += diff * w[d]

        # extract a sample
        h_x[l_window - n_lag + i] = H[-n_lag]


def sliding_transform(x, l_window, n_lag):
    """
    Sliding window Hilbert transform.

    Parameters:
    -----------
    x : 1darray
        An array of time samples.
    l_window : int
        The window length.
    n_lag : int
        To avoid the Hilbert transform's edge effects, lag `n_lag` samples behind the newest time sample.

    Returns:
    --------
    h_x : 1darray
        The sliding window Hilbert transform of `x`.
        The first `l_window` - `n_lag` + 1 samples are np.nan and the last `n_lag` - 1 samples are np.nan.
    """
    W = np.zeros(l_window, dtype=complex)
    if l_window % 2 == 0:
        W[0] = 1
        W[l_window // 2] = 1
        W[1 : l_window // 2] = 2
    else:
        W[0] = 1
        W[1 : (l_window + 1) // 2] = 2

    w = scipy.fft.ifft(W)
    w = np.roll(scipy.fft.ifft(W), -1)

    # initial Hilbert transform
    H = scipy.fft.fft(x[:l_window])
    H = scipy.fft.ifft(H * W)

    # slide
    h_x = np.zeros(x.shape, dtype=complex) * np.nan
    _sliding_transform_jit(H, w, x, h_x, l_window, n_lag)

    return h_x


if __name__ == "__main__":
    # test transform
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
    print("transform: test passed.")

    # test sliding_transform
    for n_samples, n_window, n_lag in [
        (50, 40, 5),
        (100, 30, 10),
        (143, 142, 100),
        (252, 251, 101),
        (500, 490, 40),
    ]:
        x_test = np.random.randn(n_samples)
        r1 = _sliding_transform_reference(x_test, n_window, n_lag)
        r2 = sliding_transform(x_test, n_window, n_lag)
        np.testing.assert_allclose(r1, r2)
    print("sliding_transform: test passed.")
