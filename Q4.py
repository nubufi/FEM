import numpy as np
from objects import Node
from sympy import *

x, y = symbols("x y")


class Q4:
    def __init__(self, width, height, elastic_modulus, poisson_ratio, thickness):
        self.E = elastic_modulus
        self.nu = poisson_ratio
        self.t = thickness
        self.width = width
        self.height = height
        self.a = self.width / 2
        self.h = self.height / 2
        self.A = 4 * self.a * self.h

    def B_matrix(self):
        B1 = y - self.h
        B2 = self.h - y
        B3 = self.h + y
        B4 = -self.h - y

        g1 = x - self.a
        g2 = -self.a - x
        g3 = self.a + x
        g4 = self.a - x

        B = Matrix(
            [
                [B1, 0, B2, 0, B3, 0, B4, 0],
                [0, g1, 0, g2, 0, g3, 0, g4],
                [g1, B1, g2, B2, g3, B3, g4, B4],
            ]
        )

        return B / self.A

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
        y_integration = integrate(mult, (y, -self.h, self.h))
        x_integration = integrate(y_integration, (x, -self.a, self.a))
        K = np.array(x_integration).astype(np.float64)
        return K * self.t


lst = Q4(35, 20, 2e6, 0.3, 0.5)
print(lst.K_matrix())
