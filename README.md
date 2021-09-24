##### Source code for the paper: [Vanishing Boosted Weights (VBW): A Consistent Algorithm to Learn Interpretable Rules](https://www.sciencedirect.com/science/article/pii/S0167865521003081)

![](figureIntuition.png?raw=true "figureIntuition")

This Python package allows to reproduce the results presented in the paper "Vanishing Boosted Weights: a Consistent Algorithm to Learn Interpretable Rules". The potential users can run the new method on their own data.

The user can see the user manual by executing ```python3 manager.py -h```.

```
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
```

The list of learning approaches are:<br>
GBoost : Gradient Boosting Decition Trees<br>
CatB : CatBoost Classifier<br>
LightGBM : Light Gradient Boosting Machine<br>
GOSS : Gradient-Based One Side Sampling<br>
VBW : Vanishing Boosted Weights<br>
Averaged : Corrective Federated Averaging VBW<br>


Below is an example of using the package:

```
python3 manager.py -a VBW GBoost GOSS Averaged -f 10 -e 1 10 25 75 90 -d Examples -p 15 
```

                       
A dataset is a text file comprising of all the samples. Each sample is represented by a line of features separated by space. The last column is the class which is either 1 or -1. 
