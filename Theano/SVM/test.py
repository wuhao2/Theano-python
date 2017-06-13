import numpy as np
import matplotlib.pyplot as plt

# x = np.random.rand(100)
# y = np.random.rand(100)
# t = np.arange(100)
# plt.scatter(x, y, c=t)
# plt.show()


# x = np.random.rand(100)
# y = x
# t = x
# plt.scatter(x, y, c=t)
# plt.show()


x = np.arange(100)
y = x
t = x
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(x, y, c=t, cmap='viridis')
ax2.scatter(x, y, c=t, cmap='viridis_r')
plt.show()