import dnest4.classic as dn4
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt

# Set up fonts
plt.rc("font", size=14, family="serif", serif="Computer Sans")
plt.rc("text", usetex=True)

# Set rng seed
rng.seed(3)

# x coordinate
x = np.linspace(5.0, 50.0, 1001)

# Background level
b = 1.0

# Create figure
plt.figure(1)
for i in range(0, 3):
    # Control point parameters
    n = rng.randn(4)

    # The four control points
    xx = np.array([x.min(), 10.0, 40.0, x.max()])
    yy = b * np.exp(n)

    # Plot the background component shape
    plt.plot(xx, yy, "ko-", alpha=0.2)

plt.xlabel("$x$", fontsize=16)
plt.ylabel("$y$", fontsize=16)
plt.title("Background component")
plt.savefig("figures/background.pdf", bbox_inches="tight")
plt.show()

# Load some prior samples
sample = dn4.my_loadtxt("figures/sample.txt")
indices = dn4.load_column_names("figures/sample.txt")["indices"]

x = dn4.my_loadtxt("../src/easy_data.txt")[:,0]
start = indices["wide[0]"]
end   = indices["wide[1000]"] + 1

for i in range(0, 10):
    y = sample[i, start:end]
    y /= y.max()
    plt.plot(x, y, "-")
plt.show()

