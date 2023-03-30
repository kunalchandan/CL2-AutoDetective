"""
Simulator Module
"""
# Environment imports
import sys
from pathlib import Path

import time

from typing import Dict, Type

# Math imports
import math

# Carla
from dotmap import DotMap
from navigation.agent import Agent

ROOT = Path(Path(__file__).parent.resolve())
sys.path.append(str(Path(ROOT, 'VerifAI', 'src')))
sys.path.append(str(Path(ROOT, 'VerifAI', 'src', 'verifai', 'simulators', 'carla', 'agents')))
sys.path.append(str(Path(ROOT, 'carla', 'PythonAPI', 'carla')))
from verifai.simulators.carla.client_carla import ClientCarla
from verifai.simulators.carla.carla_task import carla_task
from verifai.simulators.carla.carla_world import Vehicle

from verifai.simulators.carla.agents.brake_agent import BrakeAgent
from verifai.simulators.carla.agents.pid_agent import PIDAgent
from verifai.simulators.carla.agents.overtake_agent import OvertakeAgent

from ego_agent import EgoAgent

from carla import Transform, Rotation, Location # pylint: disable=no-name-in-module

AGENTS : Dict[str, Type[Agent]] = {'BrakeAgent': BrakeAgent, 'PIDAgent': PIDAgent, 'OvertakeAgent': OvertakeAgent, 'EgoAgent': EgoAgent}
WORLD_MAP = 'Town04'

# Falsifier (not CARLA) params
PORT = 8888
BUFSIZE = 4096

simulation_data = DotMap()
simulation_data.port = PORT
simulation_data.bufsize = BUFSIZE


class CustomCarlaTask(carla_task):
    """Custom Carla Task"""
    def __init__(self, # pylint: disable=too-many-arguments
                 n_sim_steps=250,
                 display_dim=(1280,720),
                 carla_host='127.0.0.1',
                 carla_port=2000,
                 carla_timeout=4.0,
                 world_map='Town04'):
        super().__init__(
            n_sim_steps=n_sim_steps,
            display_dim=display_dim,
            carla_host=carla_host,
            carla_port=carla_port,
            carla_timeout=carla_timeout,
            world_map=world_map
        )
        self.objects = None
        self.ego_vehicle : Vehicle = Vehicle(None, None)

    def snap_to_ground(self, location):
        '''Mutates @location to have the same z-coordinate as the nearest waypoint.'''
        waypoint = self.world.world.get_map().get_waypoint(location)
        location.z = waypoint.transform.location.z + 1
        return location

    def use_sample(self, sample):
        self.objects = sample.objects
        for obj in sample.objects:
            spawn = Transform(self.snap_to_ground(Location(x=obj.position[0],
                                                           y=-obj.position[1], z=1)),
                              Rotation(yaw=-obj.heading * 180 / math.pi - 90))
            attrs = {}
            if 'color' in obj._fields:
                color = str(int(obj.color.r * 255)) + ',' \
                    + str(int(obj.color.g * 255)) + ',' + str(int(obj.color.b * 255))
                attrs['color'] = color
            if 'blueprint' in obj._fields:
                attrs['blueprint_filter'] = obj.blueprint
            agent : Type[Agent] = PIDAgent
            if 'agent' in obj._fields:
                agent = AGENTS[obj.agent]
            if obj.type in ['Vehicle', 'Car', 'Truck', 'Bicycle', 'Motorcycle']:
                if obj is sample.objects[0]:
                    self.ego_vehicle = self.world.add_vehicle(AGENTS['EgoAgent'],
                                       spawn=spawn,
                                       has_collision_sensor=True,
                                       has_lane_sensor=False,
                                       ego=obj is sample.objects[0],
                                       **attrs)
                else:
                    self.world.add_vehicle(agent,
                                       spawn=spawn,
                                       has_collision_sensor=False,
                                       has_lane_sensor=False,
                                       ego=obj is sample.objects[0],
                                       **attrs)

            elif obj.type == 'Pedestrian':
                self.world.add_pedestrian(spawn=spawn, **attrs)
            elif obj.type in ['Prop', 'Trash', 'Cone']:
                self.world.add_prop(spawn=spawn, **attrs)
            else:
                print('Unsupported object type:', obj.type)


    def trajectory_definition(self):
        # Get speed of collision as proportion of target speed.
        collision = [(c[0], c[1]) for c in self.ego_vehicle.collision_sensor.get_collision_speeds()] # type: ignore[union-attr]

        # MTL doesn't like empty lists.
        if not collision:
            collision = [(0,0)]
        # print(collision)
        traj = {
            'collision': collision,
            }
        return traj


def main():
    """Main simulator function"""
    # Note: The world_map param below should correspond to the MapPath
    # specified in the scenic file. E.g., if world_map is 'Town01',
    # the MapPath in the scenic file should be the path to Town01.xodr.
    simulation_data.task = CustomCarlaTask(world_map=WORLD_MAP)

    print(f"Simulation Task Defined as : {WORLD_MAP}")
    client_task = ClientCarla(simulation_data)
    while True:
        try:
            while client_task.run_client():
                pass
        except RuntimeError:
            print("Falsifier server does not seem to be running. Trying again in 2 seconds")
            time.sleep(2.0)
    print('End of all simulations.')


if __name__ == "__main__":
    main()
