import numpy as np
from sympy import *

x, y = symbols("x y")


class LST:
    def __init__(self, width, height, elastic_modulus, poisson_ratio, thickness):
        self.E = elastic_modulus
        self.nu = poisson_ratio
        self.t = thickness
        self.a = width
        self.h = height
        self.A = width * height * 0.5

    def B_matrix(self):
        B1 = -3 * self.h + 4 * self.h * x / self.a + 4 * y
        B2 = -self.h + 4 * self.h * x / self.a
        B3 = 0
        B4 = 4 * y
        B5 = -4 * y
        B6 = 4 * self.h - 8 * self.h * x / self.a + 4 * y

        g1 = -3 * self.a + 4 * x + 4 * self.a * y / self.h
        g2 = 0
        g3 = -self.a + 4 * self.a * y / self.h
        g4 = 4 * x
        g5 = -4 * self.a - 4 * x - 8 * y * self.a / self.h
        g6 = -4 * x

        B = Matrix(
            [
                [B1, 0, B2, 0, B3, 0, B4, 0, B5, 0, B6, 0],
                [0, g1, 0, g2, 0, g3, 0, g4, 0, g5, 0, g6],
                [g1, B1, g2, B2, g3, B3, g4, B4, g5, B5, g6, B6],
            ]
        )

        return B * 0.5 / self.A

    def D_matrix(self):
        constant = self.E / (1 - self.nu ** 2)
        D = (
            np.array([[1, self.nu, 0], [self.nu, 1, 0], [0, 0, (1 - self.nu) / 2]])
            * constant
        )

        return D

    def K_matrix(self):
        B = self.B_matrix()
        D = self.D_matrix()
        mult = (B.T) * D * B
        y_integration = integrate(mult, (y, 0, self.h))
        x_integration = integrate(y_integration, (x, 0, self.a))
        K = np.array(x_integration).astype(np.float64)
        return K * self.t


lst = LST(16, 12, 2e6, 0.3, 0.5)
lst.K_matrix()
print("K = 1e6 * ")
print(lst.K_matrix())
