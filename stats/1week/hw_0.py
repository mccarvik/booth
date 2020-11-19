
import numpy as np

z = np.array([2, -2, 3, -3])

# 1ai
print(z.sum())
# 1aii
z2 = (z - z.mean())**2
print(z2.sum())
# 1aii
# sample var has 1 degree of freedom
print(z2.var(ddof=1))

# 1b
# written

# 1c

X = np.array([1,2,3,4])
Y = np.array([3, 6, 9, -2])
xx = X - X.mean()
yy = Y - Y.mean()
print((xx * yy).sum())
print((xx * Y).sum())

import scipy.stats
print(scipy.stats.norm(-10, 25).cdf(-10))
print(1 - scipy.stats.norm(-10, 25).cdf(-20))
perc = scipy.stats.norm(-10, 5).cdf(-12)
print(perc)
print(scipy.stats.norm.ppf(perc))

std = 5
n = 100
mean = 50
sample = 51
calc = (sample - mean) / (std / np.sqrt(n))
print(calc)
print(1 - scipy.stats.norm.cdf(calc))
sample = 48
calc = (sample - mean) / (std / np.sqrt(n))
print(calc)
print(1 - scipy.stats.norm.cdf(calc))
sample = 49.5
calc = (sample - mean) / (std / np.sqrt(n))
print(calc)
print(1 - scipy.stats.norm.cdf(calc))
sample = 49.5
calc = (sample - mean) / (std / np.sqrt(n))
print(calc)
print((1 - 2* scipy.stats.norm.cdf(calc)))

