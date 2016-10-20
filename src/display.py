import dnest4.classic as dn4
import numpy as np
import matplotlib.pyplot as plt

def display(data_file="50% glass .02step3secdwell.txt"):
    """
    Function to load and plot the output of a run.
    """
    data = dn4.my_loadtxt(data_file)
    posterior_sample = dn4.my_loadtxt("posterior_sample.txt")
    temp = dn4.load_column_names("posterior_sample.txt")
    indices = temp["indices"]

    for i in range(0, posterior_sample.shape[0]):
        plt.clf()

        # Plot the data
        plt.plot(data[:,0], data[:,1], "ko", markersize=3)
        plt.hold(True)

        # Extract the model curve
        start = indices["model_curve[0]"]
        end = start + data.shape[0]
        model = posterior_sample[i, start:end]

        # Plot the data
        plt.plot(data[:,0], model, "g", linewidth=2)
        plt.xlabel("$x$", fontsize=16)
        plt.ylabel("$y$", fontsize=16)
        plt.show()

if __name__ == "__main__":
    display()

