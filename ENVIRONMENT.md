# General documentation for me on how I set up the environment for this god-forsaken project

## Why not poetry
I hate poetry with a passion now.

It takes well over 4 hours to tell me that a set of contraints just isn't possible. I want to kill myslef after that ordeal that has consumed well over 3 days of my life.

Installing pipx

```
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```


Installing virtualenv

```
pipx install virtualenv
```

Setup the virtual environment
```
virtualenv venv
```


Activate env with:
```
source venv/bin/activate
```


Install packages
```
pip install numpy dill dotmap networkx
pip install scenic # installs a whole bunch of packages
pip install carla
pip install pandas pyyaml
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install psutil
pip install tqdm
pip install seaborn
pip install progressbar2
pip install metric-temporal-logic
pip install decorator
pip install scikit-learn # sklearn is old repo
pip install kmodes
pip install statsmodels
```


Reasoning:
```
scenic 2.0.0 requires attrs<20.0.0,>=19.3.0
# Above is not true
# observing the pyptoject.toml for scenic has
# 	"attrs >= 19.3.0",
metric-temporal-logic 0.4.1 requires attrs<23,>=22
```


Dowloading the Carla zip is too big, I only have a few GB to play with on this computer. 
I will reuse the carla dowload in the `e5_5044` user's desktop.
