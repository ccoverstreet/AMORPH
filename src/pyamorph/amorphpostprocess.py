# Uses actual DNest4 stuff if it's installed - otherwise
# uses dnest_functions.py as a fallback
#try:
#    import dnest4.classic as dn4
#except:
#    from . import dnest_functions as dn4


def amorph_postprocess():
    postprocess()

    #from . import display
    #display.display()
    display()

import copy
import numpy as np
import matplotlib.pyplot as plt

def my_loadtxt(filename, single_precision=False, delimiter=" "):
    """
    Load quickly
    """
    # Open the file
    f = open(filename, "r")

    # Storage
    results = []

    # Rows and columns
    nrow = 0
    ncol = None

    while(True):
        # Read the line and split by whitespace
        line = f.readline()
        if delimiter != " ":
            cells = line.split(delimiter)
        else:
            cells = line.split()

        # Quit when you see a different number of columns
        if ncol is not None and len(cells) != ncol:
            break

        # Non-comment lines
        if cells[0][0] != "#":
            # If it's the first one, get the number of columns
            if ncol is None:
                ncol = len(cells)

            # Otherwise, include in results
            if single_precision:
                results.append(np.array([float(cell) for cell in cells],\
                                                          dtype="float32"))
            else:
                results.append(np.array([float(cell) for cell in cells]))
            nrow += 1

    results = np.vstack(results)
    return results

def loadtxt_rows(filename, rows, single_precision=False):
    """
    Load only certain rows
    """
    # Open the file
    f = open(filename, "r")

    # Storage
    results = {}

    # Row number
    i = 0

    # Number of columns
    ncol = None

    while(True):
        # Read the line and split by whitespace
        line = f.readline()
        cells = line.split()

        # Quit when you see a different number of columns
        if ncol is not None and len(cells) != ncol:
            break

        # Non-comment lines
        if cells[0] != "#":
            # If it's the first one, get the number of columns
            if ncol is None:
                ncol = len(cells)

            # Otherwise, include in results
            if i in rows:
                if single_precision:
                    results[i] = np.array([float(cell) for cell in cells],\
                                                              dtype="float32")
                else:
                    results[i] = np.array([float(cell) for cell in cells])
            i += 1

    results["ncol"] = ncol
    return results

def load_column_names(filename):
    """
    Read the first line of the file and extract column names from it.
    Returns a dictionary with two elements:
        colnames        A list of column names
        indices         A dictionary of column indices, indexed by name
    """
    f = open(filename, "r")
    line = f.readline()
    f.close()

    names = line.replace("#", "").replace(" ", "").replace("\n", "")\
                .split(",")
    indices = {}
    for i in range(0, len(names)):
        indices[names[i]] = i

    return {"colnames": names, "indices": indices}


def logsumexp(values):
	biggest = np.max(values)
	x = values - biggest
	result = np.log(np.sum(np.exp(x))) + biggest
	return result

def logdiffexp(x1, x2):
	biggest = x1
	xx1 = x1 - biggest
	xx2 = x2 - biggest
	result = np.log(np.exp(xx1) - np.exp(xx2)) + biggest
	return result


def postprocess(temperature=1., numResampleLogX=1, plot=True, loaded=[], \
			cut=0., save=True, zoom_in=True, compression_bias_min=1., verbose=True,\
			compression_scatter=0., moreSamples=1., compression_assert=None, single_precision=False):
	if len(loaded) == 0:
		levels_orig = np.atleast_2d(my_loadtxt("levels.txt"))
		sample_info = np.atleast_2d(my_loadtxt("sample_info.txt"))
	else:
		levels_orig, sample_info = loaded[0], loaded[1]

	# Remove regularisation from levels_orig if we asked for it
	if compression_assert is not None:
		levels_orig[1:,0] = -np.cumsum(compression_assert*np.ones(levels_orig.shape[0] - 1))

	cut = int(cut*sample_info.shape[0])
	sample_info = sample_info[cut:, :]

	if plot:
		plt.figure(1)
		plt.plot(sample_info[:,0], "k")
		plt.xlabel("Iteration")
		plt.ylabel("Level")

		plt.figure(2)
		plt.subplot(2,1,1)
		plt.plot(np.diff(levels_orig[:,0]), "k")
		plt.ylabel("Compression")
		plt.xlabel("Level")
		xlim = plt.gca().get_xlim()
		plt.axhline(-1., color='g')
		plt.axhline(-np.log(10.), color='g', linestyle="--")
		plt.ylim(top=0.05)

		plt.subplot(2,1,2)
		good = np.nonzero(levels_orig[:,4] > 0)[0]
		plt.plot(levels_orig[good,3]/levels_orig[good,4], "ko-")
		plt.xlim(xlim)
		plt.ylim([0., 1.])
		plt.xlabel("Level")
		plt.ylabel("MH Acceptance")

	# Convert to lists of tuples
	logl_levels = [(levels_orig[i,1], levels_orig[i, 2]) for i in range(0, levels_orig.shape[0])] # logl, tiebreaker
	logl_samples = [(sample_info[i, 1], sample_info[i, 2], i) for i in range(0, sample_info.shape[0])] # logl, tiebreaker, id
	logx_samples = np.zeros((sample_info.shape[0], numResampleLogX))
	logp_samples = np.zeros((sample_info.shape[0], numResampleLogX))
	logP_samples = np.zeros((sample_info.shape[0], numResampleLogX))
	P_samples = np.zeros((sample_info.shape[0], numResampleLogX))
	logz_estimates = np.zeros((numResampleLogX, 1))
	H_estimates = np.zeros((numResampleLogX, 1))

	# Find sandwiching level for each sample
	sandwich = sample_info[:,0].copy().astype('int')
	for i in range(0, sample_info.shape[0]):
		while sandwich[i] < levels_orig.shape[0]-1 and logl_samples[i] > logl_levels[sandwich[i] + 1]:
			sandwich[i] += 1

	for z in range(0, numResampleLogX):
		# Make a monte carlo perturbation of the level compressions
		levels = levels_orig.copy()
		compressions = -np.diff(levels[:,0])
		compressions *= compression_bias_min + (1. - compression_bias_min)*np.random.rand()
		compressions *= np.exp(compression_scatter*np.random.randn(compressions.size))
		levels[1:, 0] = -compressions
		levels[:, 0] = np.cumsum(levels[:,0])

		# For each level
		for i in range(0, levels.shape[0]):
			# Find the samples sandwiched by this level
			which = np.nonzero(sandwich == i)[0]
			logl_samples_thisLevel = [] # (logl, tieBreaker, ID)
			for j in range(0, len(which)):
				logl_samples_thisLevel.append(copy.deepcopy(logl_samples[which[j]]))
			logl_samples_thisLevel = sorted(logl_samples_thisLevel)
			N = len(logl_samples_thisLevel)

			# Generate intermediate logx values
			logx_max = levels[i, 0]
			if i == levels.shape[0]-1:
				logx_min = -1E300
			else:
				logx_min = levels[i+1, 0]
			Umin = np.exp(logx_min - logx_max)

			if N == 0 or numResampleLogX > 1:
				U = Umin + (1. - Umin)*np.random.rand(len(which))
			else:
				U = Umin + (1. - Umin)*np.linspace(1./(N+1), 1. - 1./(N+1), N)
			logx_samples_thisLevel = np.sort(logx_max + np.log(U))[::-1]

			for j in range(0, which.size):
				logx_samples[logl_samples_thisLevel[j][2]][z] = logx_samples_thisLevel[j]

				if j != which.size - 1:
					left = logx_samples_thisLevel[j+1]
				elif i == levels.shape[0]-1:
					left = -1E300
				else:
					left = levels[i+1][0]

				if j != 0:
					right = logx_samples_thisLevel[j-1]
				else:
					right = levels[i][0]

				logp_samples[logl_samples_thisLevel[j][2]][z] = np.log(0.5) + logdiffexp(right, left)

		logl = sample_info[:,1]/temperature

		logp_samples[:,z] = logp_samples[:,z] - logsumexp(logp_samples[:,z])
		logP_samples[:,z] = logp_samples[:,z] + logl
		logz_estimates[z] = logsumexp(logP_samples[:,z])
		logP_samples[:,z] -= logz_estimates[z]
		P_samples[:,z] = np.exp(logP_samples[:,z])
		H_estimates[z] = -logz_estimates[z] + np.sum(P_samples[:,z]*logl)

		if plot:
			plt.figure(3)

			plt.subplot(2,1,1)
			plt.plot(logx_samples[:,z], sample_info[:,1], 'k.', label='Samples')
			plt.plot(levels[1:,0], levels[1:,1], 'g.', label='Levels')
			plt.legend(numpoints=1, loc='lower left')
			plt.ylabel('log(L)')
			plt.title(str(z+1) + "/" + str(numResampleLogX) + ", log(Z) = " + str(logz_estimates[z][0]))
			# Use all plotted logl values to set ylim
			combined_logl = np.hstack([sample_info[:,1], levels[1:, 1]])
			combined_logl = np.sort(combined_logl)
			lower = combined_logl[int(0.1*combined_logl.size)]
			upper = combined_logl[-1]
			diff = upper - lower
			lower -= 0.05*diff
			upper += 0.05*diff
			if zoom_in:
				plt.ylim([lower, upper])
			xlim = plt.gca().get_xlim()

		if plot:
			plt.subplot(2,1,2)
			plt.plot(logx_samples[:,z], P_samples[:,z], 'k.')
			plt.ylabel('Posterior Weights')
			plt.xlabel('log(X)')
			plt.xlim(xlim)

	P_samples = np.mean(P_samples, 1)
	P_samples = P_samples/np.sum(P_samples)
	logz_estimate = np.mean(logz_estimates)
	logz_error = np.std(logz_estimates)
	H_estimate = np.mean(H_estimates)
	H_error = np.std(H_estimates)
	ESS = np.exp(-np.sum(P_samples*np.log(P_samples+1E-300)))

	errorbar1 = ""
	errorbar2 = ""
	if numResampleLogX > 1:
		errorbar1 += " +- " + str(logz_error)
		errorbar2 += " +- " + str(H_error)

	if verbose:
		print("log(Z) = " + str(logz_estimate) + errorbar1)
		print("Information = " + str(H_estimate) + errorbar2 + " nats.")
		print("Effective sample size = " + str(ESS))

	# Resample to uniform weight
	N = int(moreSamples*ESS)
	w = P_samples
	w = w/np.max(w)
	rows = np.empty(N, dtype="int64")
	for i in range(0, N):
		while True:
			which = np.random.randint(sample_info.shape[0])
			if np.random.rand() <= w[which]:
				break
		rows[i] = which + cut

    # Get header row
	f = open("sample.txt", "r")
	line = f.readline()
	if line[0] == "#":
		header = line[1:]
	else:
		header = ""
	f.close()

	sample = loadtxt_rows("sample.txt", set(rows), single_precision)
	posterior_sample = None
	if single_precision:
		posterior_sample = np.empty((N, sample["ncol"]), dtype="float32")
	else:
		posterior_sample = np.empty((N, sample["ncol"]))

	for i in range(0, N):
		posterior_sample[i, :] = sample[rows[i]]


	if save:
		np.savetxt('weights.txt', w)
		if single_precision:
			np.savetxt("posterior_sample.txt", posterior_sample, fmt="%.7e",\
													header=header)
		else:
			np.savetxt("posterior_sample.txt", posterior_sample,\
													header=header)

	if plot:
		plt.show()

	return [logz_estimate, H_estimate, logx_samples]

def postprocess_abc(temperature=1., numResampleLogX=1, plot=True, loaded=[], \
			cut=0., save=True, zoom_in=True, compression_bias_min=1., verbose=True,\
compression_scatter=0., moreSamples=1., compression_assert=None, single_precision=False, threshold_fraction=0.8):
	if len(loaded) == 0:
		levels_orig = np.atleast_2d(my_loadtxt("levels.txt"))
		sample_info = np.atleast_2d(my_loadtxt("sample_info.txt"))
	else:
		levels_orig, sample_info = loaded[0], loaded[1]

	# Remove regularisation from levels_orig if we asked for it
	if compression_assert is not None:
		levels_orig[1:,0] = -np.cumsum(compression_assert*np.ones(levels_orig.shape[0] - 1))

	cut = int(cut*sample_info.shape[0])
	sample_info = sample_info[cut:, :]

	if plot:
		plt.figure(1)
		plt.plot(sample_info[:,0], "k")
		plt.xlabel("Iteration")
		plt.ylabel("Level")

		plt.figure(2)
		plt.subplot(2,1,1)
		plt.plot(np.diff(levels_orig[:,0]), "k")
		plt.ylabel("Compression")
		plt.xlabel("Level")
		xlim = plt.gca().get_xlim()
		plt.axhline(-1., color='g')
		plt.axhline(-np.log(10.), color='g', linestyle="--")
		plt.ylim(top=0.05)

		plt.subplot(2,1,2)
		good = np.nonzero(levels_orig[:,4] > 0)[0]
		plt.plot(levels_orig[good,3]/levels_orig[good,4], "ko-")
		plt.xlim(xlim)
		plt.ylim([0., 1.])
		plt.xlabel("Level")
		plt.ylabel("MH Acceptance")

	# Convert to lists of tuples
	logl_levels = [(levels_orig[i,1], levels_orig[i, 2]) for i in range(0, levels_orig.shape[0])] # logl, tiebreakercut
	logl_samples = [(sample_info[i, 1], sample_info[i, 2], i) for i in range(0, sample_info.shape[0])] # logl, tiebreaker, id
	logx_samples = np.zeros((sample_info.shape[0], numResampleLogX))
	logp_samples = np.zeros((sample_info.shape[0], numResampleLogX))
	logP_samples = np.zeros((sample_info.shape[0], numResampleLogX))
	P_samples = np.zeros((sample_info.shape[0], numResampleLogX))
	logz_estimates = np.zeros((numResampleLogX, 1))
	H_estimates = np.zeros((numResampleLogX, 1))

	# Find sandwiching level for each sample
	sandwich = sample_info[:,0].copy().astype('int')
	for i in range(0, sample_info.shape[0]):
		while sandwich[i] < levels_orig.shape[0]-1 and logl_samples[i] > logl_levels[sandwich[i] + 1]:
			sandwich[i] += 1

	for z in range(0, numResampleLogX):
		# Make a monte carlo perturbation of the level compressions
		levels = levels_orig.copy()
		compressions = -np.diff(levels[:,0])
		compressions *= compression_bias_min + (1. - compression_bias_min)*np.random.rand()
		compressions *= np.exp(compression_scatter*np.random.randn(compressions.size))
		levels[1:, 0] = -compressions
		levels[:, 0] = np.cumsum(levels[:,0])

		# For each level
		for i in range(0, levels.shape[0]):
			# Find the samples sandwiched by this level
			which = np.nonzero(sandwich == i)[0]
			logl_samples_thisLevel = [] # (logl, tieBreaker, ID)
			for j in range(0, len(which)):
				logl_samples_thisLevel.append(copy.deepcopy(logl_samples[which[j]]))
			logl_samples_thisLevel = sorted(logl_samples_thisLevel)
			N = len(logl_samples_thisLevel)

			# Generate intermediate logx values
			logx_max = levels[i, 0]
			if i == levels.shape[0]-1:
				logx_min = -1E300
			else:
				logx_min = levels[i+1, 0]
			Umin = np.exp(logx_min - logx_max)

			if N == 0 or numResampleLogX > 1:
				U = Umin + (1. - Umin)*np.random.rand(len(which))
			else:
				U = Umin + (1. - Umin)*np.linspace(1./(N+1), 1. - 1./(N+1), N)
			logx_samples_thisLevel = np.sort(logx_max + np.log(U))[::-1]

			for j in range(0, which.size):
				logx_samples[logl_samples_thisLevel[j][2]][z] = logx_samples_thisLevel[j]

				if j != which.size - 1:
					left = logx_samples_thisLevel[j+1]
				elif i == levels.shape[0]-1:
					left = -1E300
				else:
					left = levels[i+1][0]

				if j != 0:
					right = logx_samples_thisLevel[j-1]
				else:
					right = levels[i][0]

				logp_samples[logl_samples_thisLevel[j][2]][z] = np.log(0.5) + logdiffexp(right, left)

		logl = sample_info[:,1]/temperature

		logp_samples[:,z] = logp_samples[:,z] - logsumexp(logp_samples[:,z])

		# Define the threshold for ABC, in terms of log(X)
		threshold = threshold_fraction*levels[:,0].min()

		# Particles below threshold get no posterior weight
		logp_samples[logx_samples > threshold] = -1E300

		logP_samples[:,z] = logp_samples[:,z] + logl
		logz_estimates[z] = logsumexp(logP_samples[:,z])
		logP_samples[:,z] -= logz_estimates[z]
		P_samples[:,z] = np.exp(logP_samples[:,z])
		H_estimates[z] = -logz_estimates[z] + np.sum(P_samples[:,z]*logl)

		if plot:
			plt.figure(3)

			plt.subplot(2,1,1)
			plt.plot(logx_samples[:,z], sample_info[:,1], 'k.', label='Samples')
			plt.plot(levels[1:,0], levels[1:,1], 'g.', label='Levels')
			plt.legend(numpoints=1, loc='lower left')
			plt.ylabel('log(L)')
			plt.title(str(z+1) + "/" + str(numResampleLogX) + ", log(Z) = " + str(logz_estimates[z][0]))
			# Use all plotted logl values to set ylim
			combined_logl = np.hstack([sample_info[:,1], levels[1:, 1]])
			combined_logl = np.sort(combined_logl)
			lower = combined_logl[int(0.1*combined_logl.size)]
			upper = combined_logl[-1]
			diff = upper - lower
			lower -= 0.05*diff
			upper += 0.05*diff
			if zoom_in:
				plt.ylim([lower, upper])

			xlim = plt.gca().get_xlim()

		if plot:
			plt.subplot(2,1,2)
			plt.plot(logx_samples[:,z], P_samples[:,z], 'k.')
			plt.ylabel('Posterior Weights')
			plt.xlabel('log(X)')
			plt.xlim(xlim)

	P_samples = np.mean(P_samples, 1)
	P_samples = P_samples/np.sum(P_samples)
	logz_estimate = np.mean(logz_estimates)
	logz_error = np.std(logz_estimates)
	H_estimate = np.mean(H_estimates)
	H_error = np.std(H_estimates)
	ESS = np.exp(-np.sum(P_samples*np.log(P_samples+1E-300)))

	errorbar1 = ""
	errorbar2 = ""
	if numResampleLogX > 1:
		errorbar1 += " +- " + str(logz_error)
		errorbar2 += " +- " + str(H_error)

	if verbose:
		print("log(Z) = " + str(logz_estimate) + errorbar1)
		print("Information = " + str(H_estimate) + errorbar2 + " nats.")
		print("Effective sample size = " + str(ESS))

	# Resample to uniform weight
	N = int(moreSamples*ESS)
	w = P_samples
	w = w/np.max(w)
	rows = np.empty(N, dtype="int64")
	for i in range(0, N):
		while True:
			which = np.random.randint(sample_info.shape[0])
			if np.random.rand() <= w[which]:
				break
		rows[i] = which + cut

	sample = loadtxt_rows("sample.txt", set(rows), single_precision)
	posterior_sample = None
	if single_precision:
		posterior_sample = np.empty((N, sample["ncol"]), dtype="float32")
	else:
		posterior_sample = np.empty((N, sample["ncol"]))

	for i in range(0, N):
		posterior_sample[i, :] = sample[rows[i]]


	if save:
		np.savetxt('weights.txt', w)
		if single_precision:
			np.savetxt("posterior_sample.txt", posterior_sample, fmt="%.7e")
		else:
			np.savetxt("posterior_sample.txt", posterior_sample)

	if plot:
		plt.show()

	return [logz_estimate, H_estimate, logx_samples]




def diffusion_plot():
	"""
	Plot a nice per-particle diffusion plot.
	"""

	sample_info = np.atleast_2d(my_loadtxt('sample_info.txt'))
	ID = sample_info[:,3].astype('int')
	j = sample_info[:,0].astype('int')

	ii = np.arange(1, sample_info.shape[0] + 1)

	for i in range(0, ID.max() + 1):
		which = np.nonzero(ID == i)[0]
		plt.plot(ii[which], j[which])

	plt.xlabel('Iteration')
	plt.ylabel('Level')
	plt.show()

def levels_plot():
	"""
	Plot the differences between the logl values of the levels.
	"""
	levels = my_loadtxt('levels.txt')

	plt.plot(np.log10(np.diff(levels[:,1])), "ko-")
	plt.ylim([-1, 4])
	plt.axhline(0., color='g', linewidth=2)
	plt.axhline(np.log10(np.log(10.)), color='g')
	plt.axhline(np.log10(0.8), color='g', linestyle='--')
	plt.xlabel('Level')
	plt.ylabel('$\\log_{10}$(Delta log likelihood)')
	plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml

def display():
    """
    Function to load and plot the output of a run.
    """
    f = open("config.yaml")
    config = yaml.load(f, Loader=yaml.SafeLoader)
    f.close()

    # Load the data file
    data = my_loadtxt(config["data_file"])
    x = data[:,0]
    y = data[:,1]

    # Load posterior samples etc
    posterior_sample = my_loadtxt("posterior_sample.txt")
    temp = load_column_names("posterior_sample.txt")
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

    # A data frame to store some results
    macroquantities = pd.DataFrame()

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
    plt.xlabel("$2\\theta$ (degrees)", fontsize=14)
    plt.ylabel("Intensity", fontsize=14)
    plt.show()

    # Plot the standardised residuals of the posterior mean curve.
    # Use the posterior mean of sigma0 and sigma1 for the standardisation.
    # This is not ideal - really there is a posterior distribution over
    # residuals.
    curve = model_tot/posterior_sample.shape[0]
    sd = np.sqrt(posterior_sample[:,indices["sigma0"]].mean()**2 \
                    + posterior_sample[:,indices["sigma1"]].mean()*curve)
    resid = (curve - data[:,1])/sd
    plt.plot(data[:,0], resid, "-")
    plt.xlabel("$2\\theta$ (degrees)")
    plt.ylabel("Standardised residuals (of posterior mean model curve)")
    plt.show()

    # Number of narrow components
    binwidth=0.8
    plt.hist(posterior_sample[:,11],
             bins=np.arange(0, max_num_spikes)-0.5*binwidth,
             width=binwidth, color=[0.3, 0.3, 0.3])
    plt.xlabel("Number of narrow components")
    plt.ylabel("Number of posterior samples")

    # A data frame to store some results
    macroquantities["num_narrow_peaks"] = posterior_sample[:,11]
    plt.show()

    wide_fraction = wide_integral/(spikes_integral + wide_integral)
    macroquantities["amorph/(amorph + crystal)"] = wide_fraction

    # Remove any nans before plotting
    wide_fraction = wide_fraction[~np.isnan(wide_fraction)]

    plt.hist(wide_fraction, 100, color=[0.3, 0.3, 0.3])
    plt.xlim([0, 1])
    plt.xlabel("(amorph)/(amorph + crystal)")
    print("(amorph)/(amorph + crystal) = {a} +- {b}".format(a=wide_fraction.mean(),\
           b=wide_fraction.std()))
    plt.show()

    macroquantities["wide_center_of_mass"] = wide_center_of_mass

    wide_center_of_mass = wide_center_of_mass[~np.isnan(wide_center_of_mass)]
    plt.hist(wide_center_of_mass, 100, color=[0.3, 0.3, 0.3])
    plt.xlabel("Center of mass of wide component")
    print("Center of mass = {a} +- {b}".format(a=wide_center_of_mass.mean(),
           b=wide_center_of_mass.std()))
    plt.show()

    macroquantities["wide_halfwidth"] = wide_width
    wide_width = wide_width[~np.isnan(wide_width)]
    plt.hist(wide_width, 100, color=[0.3, 0.3, 0.3])
    plt.xlabel("Half-width of wide component")
    print("Half-width = {a} +- {b}".format(a=wide_width.mean(),
           b=wide_width.std()))
    plt.show()

    macroquantities["wide_skewness"] = wide_skewness
    wide_skewness = wide_skewness[~np.isnan(wide_skewness)]
    plt.hist(wide_skewness, 100, color=[0.3, 0.3, 0.3])
    plt.xlabel("Skewness of wide component")
    print("Skewness = {a} +- {b}".format(a=wide_skewness.mean(),
           b=wide_skewness.std()))
    plt.show()

    macroquantities["wide_nongaussianity"] = wide_nongaussianity
    wide_nongaussianity = wide_nongaussianity[~np.isnan(wide_nongaussianity)]
    plt.hist(wide_nongaussianity, 100, color=[0.3, 0.3, 0.3])
    plt.xlabel("Nongaussianity of wide component")
    plt.xlim(0.0)
    print("Nongaussianity = {a} +- {b}".format(a=wide_nongaussianity.mean(),
           b=wide_nongaussianity.std()))
    plt.show()

    print("Saving macroquantities.csv")
    macroquantities.index = range(1, posterior_sample.shape[0]+1)
    macroquantities.to_csv("macroquantities.csv",
                           index_label="Sample number")





if __name__ == "__main__":
    amorph_postprocess()
