# General documentation for me on how I set up the environment

## Why not poetry
I hate poetry with a passion now.

It takes well over 4 hours to tell me that a set of contraints just isn't possible. It has consumed well over 3 days of my life.

Installing pipx

```
python3.7 -m pip install --user pipx
python3.7 -m pipx ensurepath
```


Installing virtualenv

```
pipx install virtualenv
```

Setup the virtual environment
```
virtualenv venv --python python3.7
```


Activate env with:
```
source venv/bin/activate
```


Install packages
```
pip install numpy dill dotmap networkx
pip install scenic # installs a whole bunch of packages
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

I hate poetry, it takes well over 15000 seconds ~= 4 HOURS to create, initialize and resolve dependancy issues for environment and it doesn't even work 😭😭😭😭


~~Dowloading the Carla zip is too big, I only have a few GB to play with on this computer.
I will reuse the carla dowload in the `e5_5044` user's desktop.~~

I have become greedy.
Once the carla folder has been extracted to `/media/e5_5044/OSDisk/carla/` as indicated in the (`./README.md`)[./README.md] we must install the wheel for carla into the virtual environment that we have created.

```
pip install /media/e5_5044/OSDisk/carla/PythonAPI/carla/dist/carla-0.9.14-cp37-cp37m-manylinux_2_27_x86_64.whl
```

Now to run the Carla UE Engine

```
cd /media/e5_5044/OSDisk/carla

./CarlaUE4.sh -RenderOffScreen
```

## Optional

```
pip install pylint[spelling]
pip install bandit
pip install mypy
```

Install stubs for mypy
```
mypy --install-types
mkdir stubs
cd stubs
stubgen -m carla
mv out/carla/ ./
```

Run mypy with:
```
mypy --config-file pyproject.toml simulator.py
```

Or use pylint
```
pylint --rcfile .pylintrc simulator.py
```