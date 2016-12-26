import dnest4.classic as dn4
import numpy as np
import matplotlib.pyplot as plt

def display():
    """
    Function to load and plot the output of a run.
    """
    f = open("run_data.txt")
    data_file = f.read()
    f.close()

    data = dn4.my_loadtxt(data_file)
    posterior_sample = dn4.my_loadtxt("posterior_sample.txt")
    temp = dn4.load_column_names("posterior_sample.txt")
    indices = temp["indices"]
    wide_integral = np.zeros(posterior_sample.shape[0])
    spikes_integral = np.zeros(posterior_sample.shape[0])
    max_num_spikes = int(posterior_sample[0, indices["max_num_spikes"]])

    for i in range(0, posterior_sample.shape[0]):

        if i==0:
            plt.clf()

            # Plot the data
            plt.plot(data[:,0], data[:,1], "ko", markersize=3, alpha=0.2)
            plt.hold(True)

        # Extract the wide component
        start = indices["wide_component[0]"]
        end = start + data.shape[0]
        wide_component = posterior_sample[i, start:end]

        # Extract the spikes component
        start = indices["the_spikes[0]"]
        end = start + data.shape[0]
        the_spikes = posterior_sample[i, start:end]

        # Extract the model curve
        start = indices["model_curve[0]"]
        end = start + data.shape[0]
        model = posterior_sample[i, start:end]

        wide_integral[i] = np.sum(wide_component)
        spikes_integral[i] = np.sum(the_spikes)

        # Plot the model
        if i < 50:
            plt.plot(data[:,0], model, "g", linewidth=2, alpha=0.1)
            plt.plot(data[:,0], wide_component, "b", linewidth=2, alpha=0.1)
            plt.plot(data[:,0], the_spikes, "r", linewidth=2, alpha=0.1)
            plt.plot(data[:,0], posterior_sample[i, 0]*np.ones(len(model)),\
                                                "y", linewidth=2, alpha=0.1)
            plt.ylim(0)

    plt.xlabel("$x$", fontsize=16)
    plt.ylabel("$y$", fontsize=16)
    plt.show()

    wide_fraction = wide_integral/(spikes_integral + wide_integral)
    plt.hist(wide_fraction, 100, color=[0.8, 0.8, 0.8])
    plt.xlim([0, 1])
    plt.xlabel("(wide component flux)/(total spike flux + wide component flux)")
    print("Wide fraction = {a} +- {b}".format(a=wide_fraction.mean(),\
           b=wide_fraction.std()))
    plt.show()

if __name__ == "__main__":
    display()

