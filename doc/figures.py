import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt

# Set rng seed
rng.seed(3)

# x coordinate
x = np.linspace(5.0, 50.0, 1001)

# Background level
b = 1.0

# Create figure
plt.figure(1)
plt.hold(True)

for i in range(0, 3):
    # Control point parameters
    n = rng.randn(4)

    # The four control points
    xx = np.array([x.min(), 10.0, 40.0, x.max()])
    yy = b * np.exp(n)

    # Plot the background component shape
    plt.plot(xx, yy, "ko-", alpha=0.2)

plt.xlabel("$x$", fontsize=20)
plt.ylabel("$y$", fontsize=20)
plt.title("Background component", fontsize=14)
plt.savefig("figures/background.pdf", bbox_inches="tight")
plt.show()

