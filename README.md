# SpectrumSurveyingMLSP

The Python code in this repository implements the simulations and plots the figures described in the paper “AERIAL SPECTRUM SURVEYING: RADIO MAP ESTIMATION WITH AUTONOMOUS UAVS” by Daniel Romero, Raju Shrestha, Yves Teganya, Sundeep Prabhakar Chepuri.

#### Requirements: Python 3.6 or later



# Guidelines
First, download all the files and folders from this repository. Then, the simulations can be executed by running the file `run_experiment.py`from the command prompt. 
One needs to provide the experiment number (e.g. 1002) as an argument while executing the file `run_experiment.py`to select the simulation you want to run. 
The experiments reproducing different figures in the paper are organized in the methods located in the file `Experiments/route_planning_experiments.py`.
The comments before each method indicate which figure(s) on the paper it generates.
For example, to run the experiment no 1002 in the `Experiments/route_planning_experiments.py`, in the command prompt, execute the command `$ python run_experiment.py 1002`.

For more information about the simulation environment, please check [here](https://github.com/fachu000/GSim-Python).


# Citation
If our code is helpful in you research or work, please cite our paper:
........
