# General documentation for me on how I set up the environment

## Why not poetry
I hate poetry with a passion now.

It takes well over 4 hours to tell me that a set of contraints just isn't possible. It has consumed well over 3 days of my life.

## Using Pip, VirtualEnv
Installing pipx

```bash
python3.7 -m pip install --user pipx
python3.7 -m pipx ensurepath
```


Installing virtualenv

```bash
pipx install virtualenv
```

Setup the virtual environment
```bash
virtualenv venv --python python3.7
```


Activate env with:
```bash
source venv/bin/activate
```

## Using Packages

Install packages
```bash
pip install numpy dill dotmap networkx

# Scenic Dependancies
pip install antlr4-python3-runtime
pip install attrs
pip install importlib_metadata
pip install mapbox_earcut
pip install matplotlib
pip install opencv
pip install pillow
pip install pygame
pip install scipy
pip install shapely

pip install frozendict
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

# Local dependancies
pip install sqlalchemy
pip install psycopg2-binary
pip install dash dash-daq
pip install plotly
pip install gunicorn
```

I hate poetry, it takes well over 15000 seconds ~= 4 HOURS to create, initialize and resolve dependancy issues for environment and it doesn't even work 😭😭😭😭


~~Dowloading the Carla zip is too big, I only have a few GB to play with on this computer.~~
~~I will reuse the carla dowload in the `e5_5044` user's desktop.~~

I have become greedy.
Once the carla folder has been extracted to `/media/e5_5044/OSDisk/carla/` as indicated in the (`./README.md`)[./README.md] we must install the wheel for carla into the virtual environment that we have created.

```bash
pip install /media/e5_5044/OSDisk/carla/PythonAPI/carla/dist/carla-0.9.14-cp37-cp37m-manylinux_2_27_x86_64.whl
```

Now to run the Carla UE Engine

```bash
cd /media/e5_5044/OSDisk/carla

./CarlaUE4.sh -RenderOffScreen
```

## ~~Optional~~ Linting and Type Checking

```bash
pip install pylint[spelling]
pip install bandit
pip install mypy
pip install ipython
```

Install stubs for mypy
```bash
mypy --install-types
mkdir stubs
cd stubs
stubgen -m carla
mv out/carla/ ./
```

Run mypy with:
```bash
mypy --config-file pyproject.toml simulator.py
```

Or use pylint
```bash
pylint --rcfile .pylintrc simulator.py
```

## Optional Coloured Traceback

Install coloured output packages
```bash
pip install colored-traceback
pip install colorama
```

Run with
```bash
python -m colored_traceback AutoDetective.py
```