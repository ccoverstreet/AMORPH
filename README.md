AMORPH
========

(c) 2016–2018 Brendon J. Brewer and Michael Rowe

AMORPH is free software licenced under the GNU General Public License,
version 3. See the LICENSE file for details.

Steps to installing and Operating:
==================================
The entire repository, including C++ and Python source code,
and a precompiled executable file for Microsoft Windows (AMORPH.exe),
is hosted on Bitbucket at the following URL:
https://bitbucket.org/eggplantbren/amorph

This download includes the source code, Python scripts for viewing results,
example datasets, and the Windows executable file (AMORPH.exe), which is
the easiest way of using AMORPH.
The Python scripts make use of the packages Numpy and matplotlib, and has only
been tested under Python 3. Anaconda (https://www.anaconda.com/download/) is a
convenient distribution of Python which comes with scientific packages
pre-installed.

The AMORPH program can be installed anywhere on the computer. All data files to
be analysed by AMORPH need to be in .txt file format, space delimited, with no
headers (i.e., just two columns of numbers). For simplicity, text files for
processing should be located in the same folder as the AMORPH program.
To configure a run, edit the file `config.yaml` to specify all the details of
the data file and other things you might want to tweak.
Upon executing AMORPH.exe, input the file name including the .txt extension.
Four test data sets are provided in the download
(0%glass_rhyolite_Emperyon.txt; 50% glass.02step3secdwell.txt;
90%_glass_basalt_Emperyon.txt; easy_data.txt). The program is set to run until
10,000 saved parameter sets have been generated. Outputs may be viewed at any
point, however, closing the program before reaching  10,000 will reduce the
accuracy of the final calculations. After running for a while, the output can
be viewed by running the Python script showresults.py:

python showresults.py

Automatically generated figures may be closed or saved. Upon closing outputted
figures, numerical results will be displayed in Anaconda prompt window,
and a CSV file of (some) output will be written to the disk.
Important: make sure to save the results or copy the output file before
starting another run as results will otherwise be overwritten.

Recommendations for Use
=======================
Based on experimentation, several recommendations for optimization of use and
accuracy may be suggested to potential users. First, the time of the analysis
is dependent on the number of data points to be analyzed. Thus to optimize
analysis time, we recommend an XRD instrument step of 0.02 degrees/step.
While coarser steps will reduce analysis time, the peak broadening associated
with it may reduce precision of the fitting. We also recommend reducing the
total scan range to between 10-40 degrees (2theta). This has two advantages;
1) it will reduce the number of data points for processing and 2) outside this
range the X-ray diffraction pattern is dominated by only the crystalline
componentry and thus incorporation of higher 2-theta values skews measured
results to higher measured crystallinities.

Background positions are currently optimized for the use of CuKa x-ray sources,
with linear fits between 10-40° 2theta as described in the manuscript. For
other X-ray sources, it may be necessary to adjust these fixed points to
provide the best fit to the diffraction patterns. This may be modified by
changing the values in control_points.in (which can be opened in any text
editor). For some analyses, typically when more data points are analysed
(5-50 degrees 2theta) or for highly crystalline materials, the numerical
settings of the DNest4 sampler are not optimal for obtaining useful results,
and need to be made more conservative (making the run slower).
This is typically manifested in Python outputs as pale colours in the model
curves in the output plots (showing hardly any samples from the posterior)
and a low number of data points for subsequent histograms.
The settings can be modified in these instances by opening the OPTIONS file
in any text editor. Parameter values on line 4 (new level interval) and line 8
(Backtracking scale length) can each independently be doubled. Note, while this
will improve model outputs, it significantly lengthens the time of analysis.
Therefore we only recommend changing one of these two parameters at a time.
