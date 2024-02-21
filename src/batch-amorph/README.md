# Batch AMORPH

This script is used for running AMORPH on a set of files utilizing the same config. The script will automatically created needed run directories and copy necessary files for AMORPH.

The script expects all data files to be in a `data` directory. The folder structure should look like the list below:

- YourProject
    - `/data`
        - data files (columns of x and y data)
            - Make sure impurity peaks are removed from sample (ex. molybdenum)
    - `config.yaml`
    - `OPTIONS`
    - `batch.py`

- **IMPORT CONFIGURATION POINTS**
    - The `config.yaml` file should have `$FILENAME` as the argument 
        - This enables the script to automatically insert the correct filename
    - Remove impurity peaks prior that you do not included in amorphous fraction analysis
    - Your control points should be set accordingly
        - Put them at minima between the amorphous bands -- or -- subtract you background prior
    - For ensure that `left_edge` and `right_edge` allow for amorphous bands to be placed successfully. When in doubt, just use 0.0 and 1.0 respectively
    - Use a minimal number of narrow and wide peaks to fit your data. You can include more if you desire, but large numbers of peaks greatly reduces program performance.


**THERE WILL EVENTUALLY BE A BATCH POST-PROCESSING SCRIPT**

- Just need to write a script. Maybe a good undergrad project to work on for learning Python.
