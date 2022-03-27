from scipy.integrate import quadrature
from sympy import *

x, y = symbols("x y")
f = 2 * x ** 2 + 3 * x * y + 4 * y ** 2

i1 = integrate(f, (x, -1, 1))
i2 = integrate(i1, (y, -1, 1))

print(i2)
