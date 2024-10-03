from casadi import cos, sin
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi


class FourierSeries:
    def __init__(self):
        self.p = 1  # period value

    # function that computes the real fourier couples of coefficients (a0, 0), (a1, b1)...(aN, bN)
    def compute_real_fourier_coeffs(self, x, y, n):
        result = []
        for i in range(n + 1):
            an = (2.0 / self.p) * spi.trapezoid(y * np.cos(2 * np.pi * i * x / self.p), x)
            bn = (2.0 / self.p) * spi.trapezoid(y * np.sin(2 * np.pi * i * x / self.p), x)
            result.append((an, bn))
        return np.array(result)

    # function that computes the real form Fourier series using an and bn coefficients
    def fit_func_by_fourier_series_with_real_coeffs(self, x, ab, mode="numpy"):
        result = 0.0
        a = ab[:, 0]
        b = ab[:, 1]
        if mode == "numpy":
            for n in range(0, len(ab)):
                if n > 0:
                    result += a[n] * np.cos(2.0 * np.pi * n * x / self.p) + b[n] * np.sin(2.0 * np.pi * n * x / self.p)
                else:
                    result += a[0] / 2.0
            return result
        elif mode == "casadi":
            for n in range(0, len(ab)):
                if n > 0:
                    result += a[n] * cos(2.0 * np.pi * n * x / self.p) + b[n] * sin(2.0 * np.pi * n * x / self.p)
                else:
                    result += a[0] / 2.0
            return result

    def fourier_approx(self, x, y, n):
        # AB contains the list of couples of (an, bn) coefficients for n in 1..N interval.
        ab = self.compute_real_fourier_coeffs(x, y, n)
        # y_approx contains the discrete values of approximation obtained by the Fourier series
        y_approx = self.fit_func_by_fourier_series_with_real_coeffs(x, ab)
        # plot, in the range from 0 to P, the true f(t) in blue and the approximation in red
        plt.scatter(x, y, color="blue", s=5, marker=".")
        plt.plot(x, y_approx, color="red", linewidth=1)
        plt.show()
        return y_approx
