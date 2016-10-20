import dnest4.classic as dn4
import numpy as np
import matplotlib.pyplot as plt

def display(data_file="50% glass .02step3secdwell.txt"):
    """
    Function to load and plot the output of a run.
    """
    data = dn4.my_loadtxt(data_file)

    plt.plot(data[:,0], data[:,1], "ko", markersize=3)
    plt.xlabel("$x$", fontsize=16)
    plt.ylabel("$y$", fontsize=16)
    plt.show()

