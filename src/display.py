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
    wide_integral = np.zeros(posterior_sample.shape[0])
    spikes_integral = np.zeros(posterior_sample.shape[0])
    max_num_spikes = int(posterior_sample[0, indices["max_num_spikes"]])

    for i in range(0, posterior_sample.shape[0]):

        if i==0:
            plt.clf()

            # Plot the data
            plt.plot(data[:,0], data[:,1], "ko", markersize=3)
            plt.hold(True)

        # Extract the model curve
        start = indices["model_curve[0]"]
        end = start + data.shape[0]
        model = posterior_sample[i, start:end]

        wide_integral[i] = posterior_sample[i, indices["amplitude"]]*\
                           posterior_sample[i, indices["width"]]*\
                           np.sqrt(2*np.pi)

        spikes_a = np.exp(posterior_sample\
                    [i, indices["log_amplitude[0]"]:\
                    indices["log_amplitude[0]"] + max_num_spikes])
        spikes_w = posterior_sample\
                    [i, indices["width[0]"]:\
                    indices["width[0]"] + max_num_spikes]
        spikes_integral[i] = np.sum(spikes_a*spikes_w)*np.sqrt(2*np.pi)

#        print(wide_integral[i], spikes_integral[i])

        # Plot the model
        if i < 50:
            plt.plot(data[:,0], model, "g", linewidth=2, alpha=0.1)

    plt.xlabel("$x$", fontsize=16)
    plt.ylabel("$y$", fontsize=16)
    plt.show()

    spikes_fraction = spikes_integral/(spikes_integral + wide_integral)
    plt.hist(spikes_fraction, 100, color=[0.8, 0.8, 0.8])
    plt.xlim([0, 1])
    plt.xlabel("(total spike flux)/(total spike flux + wide component flux)")
    plt.show()

if __name__ == "__main__":
    display()

