"""
Falsifier Module
"""
# Environment Imports
import sys
from pathlib import Path

# Logging imports
import logging
from logging.handlers import QueueHandler
from queue import Queue
from logging import Handler

# Falsifier Imports
from dotmap import DotMap

import data_logging

ROOT = Path(Path(__file__).parent.resolve())
sys.path.append(str(Path(ROOT, 'VerifAI', 'src')))
sys.path.append(str(Path(ROOT, 'Scenic', 'src')))

from verifai.samplers.scenic_sampler import ScenicSampler
from verifai.falsifier import mtl_falsifier

CARLA_MAP = 'Town04.xodr'
MAP_PATH = str(Path.joinpath(ROOT, 'Scenic', 'tests', 'formats', 'opendrive', 'maps', 'CARLA', CARLA_MAP))
PATH_TO_SCENIC_FILE = Path.joinpath(ROOT, 'scenarios', 'oas_scenario_06.scenic')

# Falsifier Parameters
MAX_ITERS = 10
MAX_TIME = 100
PORT = 8888
MAXREQS = 5
BUFSIZE = 4096
specification = ['G(~(collision))']



def main(log_queue: Queue):
    logger = data_logging.set_queue_logger('FalsifierLogger', log_queue)

    falsifier_params = DotMap(
        n_iters=MAX_ITERS,
        max_time=MAX_TIME,
        save_error_table=True,
        save_safe_table=True,
        error_table_path=Path.joinpath(ROOT, 'falsifier_outputs', 'error_table.csv'),
        safe_table_path=Path.joinpath(ROOT, 'falsifier_outputs', 'safe_table.csv')
    )

    server_options = DotMap(port=PORT, bufsize=BUFSIZE, maxreqs=MAXREQS)

    # sampler = ScenicSampler.fromScenario(PATH_TO_SCENIC_FILE,
    #                                     params={'map' : MAP_PATH,
    #                                             'carla_map' : CARLA_MAP})

    sampler = ScenicSampler.fromScenario(PATH_TO_SCENIC_FILE)

    falsifier = mtl_falsifier(sampler=sampler,
                              sampler_type='scenic',
                              falsifier_params=falsifier_params,
                              specification=specification,
                              server_options=server_options)


    logging.debug(f"error_table_path: {falsifier.error_table.table}")
    logging.debug(f"safe_table_path: {falsifier.safe_table.table}")


    logging.debug("Running Falsifier")
    falsifier.run_falsifier()
    logging.debug("Falsifier Ended")
