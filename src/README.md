## AMORPH Source Directory

- Executables are located in `/dist`
- To run you need the `config.yaml` and either the `OPTIONS` or `OPTIONS_AGGRESSIVE` file
    - Make sure the `dnest4_options_file` entry in the `config.yaml` folder is set to whichever OPTIONS file you download and use.
- Post-processing script is available as a single python file `pyamorph/amorphpostprocess.py` and as a python package that can be installed using `python -m pip install pyamorph-ccoverstreet`
    - To run the python module version, use `python -m pyamorph` in the directory with your AMORPH outputs post-process your data
- Feel free to raise an issue on this repository if you issues compiling/running AMORPH and the post-processing component. 
