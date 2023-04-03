from verifai.samplers.scenic_sampler import ScenicSampler
from dotmap import DotMap
from verifai.falsifier import mtl_falsifier

import csv
import pathlib

LOCAL_PATH = pathlib.Path(__file__).parent.resolve()
path_to_scenic_file = pathlib.Path.joinpath(LOCAL_PATH, 'scenarios', 'oas_scenario_06.scenic')
sampler = ScenicSampler.fromScenario(path_to_scenic_file)

MAX_ITERS = 10
MAX_TIME = 100
PORT = 8888
MAXREQS = 5
BUFSIZE = 4096
specification = ['G(~(collision))']

falsifier_params = DotMap(
    n_iters=MAX_ITERS,
    max_time=MAX_TIME,
    save_error_table=True,
    save_safe_table=True,
    error_table_path=pathlib.Path.joinpath(LOCAL_PATH, 'falsifier_outputs', 'error_table.csv'),
    safe_table_path=pathlib.Path.joinpath(LOCAL_PATH, 'falsifier_outputs', 'safe_table.csv')
)

server_options = DotMap(port=PORT, bufsize=BUFSIZE, maxreqs=MAXREQS)

falsifier = mtl_falsifier(sampler=sampler, 
                          sampler_type='scenic', 
                          falsifier_params=falsifier_params, 
                          specification=specification,
                          server_options=server_options)


print("Running Falsifier")
falsifier.run_falsifier()
print("Falsifier Ended")


print("error_table: ", falsifier.error_table.table)
print("safe_table: ", falsifier.safe_table.table)

