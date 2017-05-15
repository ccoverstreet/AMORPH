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
print("figures/background.pdf done")
plt.clf()

# Load some prior samples
sample = dn4.my_loadtxt("figures/sample.txt")
indices = dn4.load_column_names("figures/sample.txt")["indices"]

x = dn4.my_loadtxt("../src/easy_data.txt")[:,0]
start = indices["wide[0]"]
end   = indices["wide[1000]"] + 1

for i in range(0, 8):
    k = rng.randint(sample.shape[0])
    y = sample[k, start:end]
    y /= y.max()# + 0.00001
    y *= np.exp(0.1*rng.randn())
    
    plt.plot(x, y, "-", alpha=0.4)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Wide component")

plt.savefig("figures/wide_component.pdf", bbox_inches="tight")
print("figures/wide_component.pdf done")
plt.clf()

# Demonstrate skewness and nongaussianity
def properties(f):
    f /= f.sum()
    com = np.sum(f*x)
    wsq = np.sum(f*(x - com)**2)
    w = np.sqrt(wsq)
    skewness = np.sum(f*((x - com)/w)**3)
    gaussian = np.exp(-0.5*(x - com)**2/wsq)
    gaussian /= np.sum(gaussian)
    KL = np.sum(f*np.log(f/gaussian))
    return (skewness, KL)

width = 3*np.ones(len(x))
f0 = np.exp(-0.5*(x - 25)**2/width**2)

width = 3 + 1.15*(x > 25)
f1 = np.exp(-0.5*(x - 25)**2/width**2)

width = 3 + 3.05*(x > 25)
f2 = np.exp(-0.5*(x - 25)**2/width**2)

plt.plot(x, f0/f0.max(), alpha=0.6, label="Skewness = 0.0")
plt.plot(x, f1/f1.max(), alpha=0.6, label="Skewness = 0.25")
plt.plot(x, f2/f2.max(), alpha=0.6, label="Skewness = 0.50")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Skewness")
plt.legend(loc="upper left")
plt.savefig("figures/skewness.pdf", bbox_inches="tight")
print("figures/skewness.pdf done")
plt.close()

nu = 100.0
f0 = (1.0 + (x - 25.0)**2/nu/3**2)**(-0.5*(nu + 1.0))
f0 /= f0.max()

nu = 3.0
f1 = (1.0 + (x - 25.0)**2/nu/3**2)**(-0.5*(nu + 1.0))
f1 /= f1.max()

nu = 1.0
f2 = (1.0 + (x - 25.0)**2/nu/3**2)**(-0.5*(nu + 1.0))
f2 /= f2.max()

plt.figure(figsize=(9, 6))
plt.plot(x, f0, alpha=0.6, label="non-gaussianity=0")
plt.plot(x, f1, alpha=0.6, label="non-gaussianity=0.08")
plt.plot(x, f2, alpha=0.6, label="non-gaussianity=0.15")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("non-gaussianity")
plt.legend(loc="upper left")
plt.savefig("figures/nongaussianity.pdf", bbox_inches="tight")
print("figures/nongaussianity.pdf done")



