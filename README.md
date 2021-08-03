# Vanishing Boosted Weights (VBW): a Consistent Algorithm to Learn Interpretable Rules

This Python package allows to reproduce the results presented in the paper "Vanishing Boosted Weights: a Consistent Algorithm to Learn Interpretable Rules". The potential users can run the new method on their own data.

usage: manager.py [-h] [-a ALGORITHM [ALGORITHM ...]] [-f FEATURES]
                  [-e ESTIMATORS [ESTIMATORS ...]] [-d DATA] [-p PROCESS]

Vanishing Boosted Weights (VBW): A corrective fine-tuning procedure on
decision stumps.

optional arguments:
  -h, --help            show this help message and exit
  -a ALGORITHM [ALGORITHM ...], --algorithm ALGORITHM [ALGORITHM ...]
                        List of arguments (default: GBoost CatB GOSS VBW
                        LightGBM Averaged])
  -f FEATURES, --features FEATURES
                        Number of features (default: 10)
  -e ESTIMATORS [ESTIMATORS ...], --estimators ESTIMATORS [ESTIMATORS ...]
                        List of number of estimators (default: 1 5 10 25 50 75
                        100)
  -d DATA, --data DATA  Path to datasets (default: ./Examples)
  -p PROCESS, --process PROCESS
                        Number of processes (default: 4)


