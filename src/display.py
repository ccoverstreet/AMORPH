# Uses actual DNest4 stuff if it's installed - otherwise
# uses dnest_functions.py as a fallback
try:
    import dnest4.classic as dn4
except:
    import dnest_functions as dn4

import numpy as np
import matplotlib.pyplot as plt

def display():
    """
    Function to load and plot the output of a run.
    """
    f = open("run_data.txt")
    data_file = f.read()
    f.close()

    # Load the data file
    data = dn4.my_loadtxt(data_file)
    x = data[:,0]
    y = data[:,1]

    # Load posterior samples etc
    posterior_sample = dn4.my_loadtxt("posterior_sample.txt")
    temp = dn4.load_column_names("posterior_sample.txt")
    indices = temp["indices"]

    # Prepare some arrays
    wide_integral = np.zeros(posterior_sample.shape[0])
    wide_center_of_mass = np.zeros(posterior_sample.shape[0])
    wide_width = np.zeros(posterior_sample.shape[0])
    wide_skewness = np.zeros(posterior_sample.shape[0])
    wide_nongaussianity = np.zeros(posterior_sample.shape[0])

    spikes_integral = np.zeros(posterior_sample.shape[0])
    max_num_spikes = int(posterior_sample[0, indices["max_num_peaks1"]])


    # For calculating posterior means of functions
    bg_tot = np.zeros(data.shape[0])
    wide_component_tot = np.zeros(data.shape[0])
    the_spikes_tot = np.zeros(data.shape[0])
    model_tot = np.zeros(data.shape[0])

    plt.clf()
    for i in range(0, posterior_sample.shape[0]):

        # Extract the background
        start = indices["bg[0]"]
        end = start + data.shape[0]
        bg = posterior_sample[i, start:end]
        bg_tot += bg

        # Extract the wide component
        start = indices["wide[0]"]
        end = start + data.shape[0]
        wide_component = posterior_sample[i, start:end]
        wide_component_tot += wide_component

        # Extract the spikes component
        start = indices["narrow[0]"]
        end = start + data.shape[0]
        the_spikes = posterior_sample[i, start:end]
        the_spikes_tot += the_spikes

        # Extract the model curve
        start = indices["model_curve[0]"]
        end = start + data.shape[0]
        model = posterior_sample[i, start:end]
        model_tot += model

        # Compute some integrals
        wide_integral[i] = np.sum(wide_component)

        if wide_integral[i] != 0.0:
            f = wide_component / wide_integral[i] # Normalised

            wide_center_of_mass[i] = np.sum(x * f)
            wide_width[i] = np.sqrt(np.sum((x - wide_center_of_mass[i])**2*f))
            wide_skewness[i] = np.sum(f *\
                                ((x - wide_center_of_mass[i]) / wide_width[i])**3)

            # Best fitting gaussian to the wide component
            gaussian = np.exp(-0.5*(x - wide_center_of_mass[i])**2 \
                                / wide_width[i]**2)
            gaussian /= gaussian.sum()

            # Nongaussianity based on KL divergence
            wide_nongaussianity[i] = np.sum(f*np.log(f / gaussian + 1E-300))

            spikes_integral[i] = np.sum(the_spikes)
        else:
            wide_center_of_mass[i] = np.NaN
            wide_width[i] = np.NaN
            wide_skewness[i] = np.NaN
            wide_nongaussianity[i] = np.NaN

        # Plot the model. 50 posterior samples and the
        # posterior mean for each part.
        if i < 50:
            plt.plot(data[:,0], model, "g", linewidth=1, alpha=0.1)
            plt.plot(data[:,0], wide_component, "b", linewidth=1, alpha=0.1)
            plt.plot(data[:,0], the_spikes, "r", linewidth=1, alpha=0.1)
            plt.plot(data[:,0], bg, "y", linewidth=1, alpha=0.1)
            plt.ylim(0)

    # Plot the posterior means
    plt.plot(data[:,0], model_tot/posterior_sample.shape[0],
                        "g-", linewidth=3, alpha=1)
    plt.plot(data[:,0], wide_component_tot/posterior_sample.shape[0],
                        "b--", linewidth=3, alpha=1)
    plt.plot(data[:,0], the_spikes_tot/posterior_sample.shape[0],
                        "r--", linewidth=3, alpha=1)
    plt.plot(data[:,0], bg_tot/posterior_sample.shape[0],
                        "y--", linewidth=3, alpha=1)

    # Plot the data
    plt.plot(data[:,0], data[:,1], "ko", markersize=3, alpha=0.5)

    plt.xlabel("$2\\theta$ (degrees)", fontsize=16)
    plt.ylabel("Intensity", fontsize=16)
    plt.show()

    # Plot the standardised residuals of the posterior mean curve.
    # Use the posterior mean of sigma0 and sigma1 for the standardisation.
    # This is not ideal - really there is a posterior distribution over
    # residuals.
    curve = model_tot/posterior_sample.shape[0]
    sd = np.sqrt(posterior_sample[:,1820].mean()**2 \
                    + posterior_sample[:,1821].mean()*curve)
    resid = (curve - data[:,1])/sd
    plt.plot(data[:,0], resid, "o", alpha=0.7)
    plt.xlabel("$2\\theta$ (degrees)")
    plt.ylabel("Standardised residuals (of posterior mean model curve)")
    plt.show()

    # Number of narrow components
    binwidth=0.8
    plt.hist(posterior_sample[:,11], bins=np.arange(0, 301)-0.5*binwidth,
             width=binwidth, color=[0.3, 0.3, 0.3])
    plt.xlabel("Number of narrow components")
    plt.ylabel("Number of posterior samples")
    plt.show()

    # Remove any nans before plotting
    wide_fraction = wide_integral/(spikes_integral + wide_integral)
    wide_fraction = wide_fraction[~np.isnan(wide_fraction)]

    plt.hist(wide_fraction, 100, color=[0.3, 0.3, 0.3])
    plt.xlim([0, 1])
    plt.xlabel("(amporph)/(amorph + crystal)")
    print("(amporph)/(amorph + crystal) = {a} +- {b}".format(a=wide_fraction.mean(),\
           b=wide_fraction.std()))
    plt.show()

    wide_center_of_mass = wide_center_of_mass[~np.isnan(wide_center_of_mass)]
    plt.hist(wide_center_of_mass, 100, color=[0.3, 0.3, 0.3])
    plt.xlabel("Center of mass of wide component")
    print("Center of mass = {a} +- {b}".format(a=wide_center_of_mass.mean(),
           b=wide_center_of_mass.std()))
    plt.show()

    wide_width = wide_width[~np.isnan(wide_width)]
    plt.hist(wide_width, 100, color=[0.3, 0.3, 0.3])
    plt.xlabel("Width of wide component")
    print("Width = {a} +- {b}".format(a=wide_width.mean(),
           b=wide_width.std()))
    plt.show()


    wide_skewness = wide_skewness[~np.isnan(wide_skewness)]
    plt.hist(wide_skewness, 100, color=[0.3, 0.3, 0.3])
    plt.xlabel("Skewness of wide component")
    print("Skewness = {a} +- {b}".format(a=wide_skewness.mean(),
           b=wide_skewness.std()))
    plt.show()

    wide_nongaussianity = wide_nongaussianity[~np.isnan(wide_nongaussianity)]
    plt.hist(wide_nongaussianity, 100, color=[0.3, 0.3, 0.3])
    plt.xlabel("Nongaussianity of wide component")
    plt.xlim(0.0)
    print("Nongaussianity = {a} +- {b}".format(a=wide_nongaussianity.mean(),
           b=wide_nongaussianity.std()))
    plt.show()

if __name__ == "__main__":
    display()

