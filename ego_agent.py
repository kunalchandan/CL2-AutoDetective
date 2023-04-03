"""
Self Driving Agent
"""
# System Imports
import sys
from pathlib import Path

# Data Imports
from enum import IntEnum
from typing import List, Dict, Iterable

# Math Imports
import math
import numpy as np
import random

# Drive Imports
import carla

ROOT = Path(Path(__file__).parent.resolve())
sys.path.append(str(Path(ROOT, 'carla', 'PythonAPI', 'carla', 'agents')))
sys.path.append(str(Path(ROOT, 'carla', 'PythonAPI', 'carla')))
# from agents.tools.misc import distance_vehicle, draw_waypoints, compute_distance

sys.path.append(str(Path(ROOT, 'VerifAI', 'src')))
from verifai.simulators.carla.agents.pid_agent import PIDAgent
from verifai.simulators.carla.agents.pid_follow_controller import PIDFollowController


# Imports related to Object Detection Model
import torch
sys.path.append(str(Path(ROOT, 'yolov5')))
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes,
    )
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox


LOCAL_PATH = Path.absolute(Path(__file__).parent.resolve())


class LaneChange(IntEnum):
    '''Enum for lane change'''
    LEFT = 1
    RIGHT = -1


class EgoAgent(PIDAgent):
    '''Basic Agent stolen from Rishi'''
    def __init__(self, vehicle, opt_dict:Dict={}):
        super().__init__(vehicle, opt_dict)
        self.row_values : List[str] = [] # list for maintaining data related to actions taken by ego
        # Distance to maintain from other vehicles.
        self.clear_dist : float = 10.0
        self.location_pid_dict : Dict[str, float] = {
            'K_P': 1.0,
            'K_D': 0.05,
            'K_I': 1.0,
            'dt': 0.05
        }
        if opt_dict:
            if 'target_speed' in opt_dict:
                self.location_pid_dict['dt'] = 1.0 / self.target_speed
            if 'clear_dist' in opt_dict:
                self.clear_dist = opt_dict['clear_dist']
            if 'location_pid_dict' in opt_dict:
                self.location_pid_dict = opt_dict['location_pid_dict']

        self.controller = PIDFollowController(
            vehicle,
            clear_dist=self.clear_dist,
            args_lateral=self.lateral_pid_dict,
            args_longitudinal=self.longitudinal_pid_dict)
        # Magnitude of lane_state is how far vehicle is from its
        # desired lane. Negative/positive means to the left/right
        # of desired lane, respectively.
        self.lane_state = 0
        self.is_changing_lane : bool = False
        self.current_image = None
        self.calc_dist : float = 0

        self.actor : carla.libcarla.Vehicle  = self._world.get_actor(vehicle.id)
        print(f"self.actor: {self.actor}")
        #Add rgb and depth sensors to the Ego here
        #Adding RGB and Depth Camera on Ego
        self.image_sizex : int = 1280
        self.image_sizey : int = 1280
        self.fov : int = 110
        self.sensor_tick : float = 0.2
        cam_bp = self._world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x", str(self.image_sizex))
        cam_bp.set_attribute("image_size_y", str(self.image_sizey))
        cam_bp.set_attribute("fov", str(self.fov))
        cam_bp.set_attribute('sensor_tick', str(self.sensor_tick))
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,0,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        self.rgb_camera = self._world.try_spawn_actor(
            cam_bp,
            cam_transform,
            self.actor,
            carla.AttachmentType.Rigid
            )

        depth_sensor_bp = self._world.get_blueprint_library().find('sensor.camera.depth')
        depth_sensor_bp.set_attribute("image_size_x", str(self.image_sizex))
        depth_sensor_bp.set_attribute("image_size_y", str(self.image_sizey))
        depth_sensor_bp.set_attribute("fov",str(110))
        depth_sensor_bp.set_attribute('sensor_tick', '0.2')
        depth_location = carla.Location(2,0,1)
        depth_rotation = carla.Rotation(0,0,0)
        depth_transform = carla.Transform(depth_location,depth_rotation)
        self.depth_sensor = self._world.spawn_actor(
            depth_sensor_bp,
            depth_transform,
            self.actor,
            attachment_type=carla.AttachmentType.Rigid
            )

        self.rgb_image = None
        self.depth_image = None

        # Model parameters
        self.imgsz=(640, 640)
        self.weights = Path(LOCAL_PATH, 'OD_model_data', 'yolo_weights.pt')
        data = Path(LOCAL_PATH, 'OD_model_data', 'carla_data.yaml')
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45 # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.device='cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False  # use FP16 half-precision inference
        dnn=False # use OpenCV DNN for ONNX inference
        self.agnostic_nms=False  # class-agnostic NMS
        self.device1 = select_device(self.device)

        # Load model for object detection
        self.od_model = DetectMultiBackend(
            self.weights,
            dnn=dnn,
            device=self.device1,
            data=data,
            fp16=half)
        self.stride : int = self.od_model.stride
        self.names : Dict[int, str] = self.od_model.names
        self.pt : bool = self.od_model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        self.classes = self.od_model.names
        #print("Model loaded")
        self.rgb_camera.listen(self.process_rgb_sensor_data)
        self.depth_sensor.listen(self.process_depth_sensor_data)
        #print("Sensors listening")

    def get_lane_change_w(self, cur_w, lane_change):
        """
        Return waypoint corresponding to LANE_CHANGE (either LEFT or RIGHT)
        from waypoint CUR_W. If lane change illegal or unsafe, returns None.
        """
        # print('enter get_lane_change_w')
        next_w = None
        def lane_safe(w):
            return self.lane_clear(w.transform.location)[0] and \
                    self.lane_clear(w.transform.location,
                                    min_dist=10*self.clear_dist,
                                    forward=w.transform.get_forward_vector())[0]
        if lane_change is LaneChange.LEFT and cur_w.lane_change & carla.LaneChange.Left:
            # print("LEFT LANE")
            self.row_values.append("Left")
            # self.row_values.append("-")
            if lane_safe(cur_w.get_left_lane()):
                next_w = cur_w.get_left_lane()
        elif lane_change is LaneChange.RIGHT and cur_w.lane_change & carla.LaneChange.Right:
            # print("Right lane")
            # self.row_values.append("-")
            self.row_values.append("Right")
            if lane_safe(cur_w.get_right_lane()):
                next_w = cur_w.get_right_lane()
        else:
            # self.row_values.append("-")
            self.row_values.append("-")
        return next_w


    def change_lane(self, lane_change=None):
        ''' By default, picks left or right at random. If no possible lane change,
        does nothing.'''
        if lane_change:
            potential_change = [lane_change]
        else:
            potential_change = [LaneChange.LEFT, LaneChange.RIGHT]
        potential_w = []
        cur_w = self.waypoints[0]
        # print("potential change variable with value: {}".format(potential_change))
        for change in potential_change:
            next_w = self.get_lane_change_w(cur_w, change)
            if next_w:
                potential_w.append((change, next_w))
        # print(potential_w)

        if potential_w:
            lane_change, next_w = random.choice(potential_w)
            self.is_changing_lane = True
            self.lane_state += lane_change
            self.waypoints : List[carla.libcarla.Waypoint] = [cur_w, next_w]


    def lane_clear(self, location, min_dist=None, forward=None):
        '''
        Check that the lane LOCATION is in is clear of vehicles.
        Lane is clear if no vehicles within MAX_DIST away.vehicle.
        If FORWARD, only check vehicles along direction o
        If not clear, return tuple of False and blocking f FORWARD.
        '''
        def norm(vec):
            return np.sqrt(vec.x ** 2 + vec.y ** 2 + vec.z ** 2)

        def norm_dot(a, b):
            a /= norm(a)
            b /= norm(b)
            dot = a.x * b.x + a.y * b.y + a.z * b.z
            return dot

        if not min_dist:
            min_dist = self.clear_dist

        lane_id = self._map.get_waypoint(location).lane_id
        vehicles: Iterable[carla.libcarla.Vehicle] = self._world.get_actors().filter('*vehicle*')
        output = self.get_detection(self.rgb_image)
        # sprint("output is: {}".format(output))
        for o in output:
            # print("o is: {}".format(o)) #o is the list of coordinates x1, y1, x2, y2, x3, y3
            # x1,y1 coordinates of the top left pixel of bounding box
            # x2,y2 coordinates of the bottom right pixel of the bounding box
            # x3 is the prediction score, y3 - predicted class number
            x1 = o[0]
            x2 = o[2]
            y1 = o[1]
            y2 = o[3]
            pred_class = self.classes[int(o[-1])] # returns names of the objects detected in list form
            centre_x = math.floor((x1+x2)/2)
            centre_y = math.floor((y1+y2)/2)

            if pred_class != "walker":

                dist = self.compute_distance_from_depth_image(centre_x, centre_y)
                self.calc_dist = dist
                if dist == 0:
                    rgb_path = Path(LOCAL_PATH, 'zero_dist', 'rgb', f'{self.temp_image.frame}.jpg')
                    depth_path = Path(LOCAL_PATH, 'zero_dist', 'depth', f'{self.temp_depth_image.frame}.jpg')
                    self.temp_image.save_to_disk(rgb_path)
                    self.temp_depth_image.save_to_disk(depth_path)
                    print("CALCULATED DISTANCE WAS ZERO!")
                    print(f"Location = {centre_x}, {centre_y}")
                    print(f"RGB frame number: {self.temp_image.frame} and Depth image number: {self.temp_depth_image.frame}")

                    #Storing the depth image in excel sheets
                    # self.depth_file = open('file.csv', 'a')
                    # for x in range(0,1080):waypoints
                    #     temp_row = []
                    #     for y in range(0,1080):
                    #         temp_value = []
                    #         temp_value.append(self.depth_waypointsimage[x][y][:])
                    #         temp_row.append(temp_value)

                    #     self.writer = csv.writer(self.depth_file)
                    #     self.writer.writerow(temp_row)
                    # self.depth_file.close()

                    ########################################

                    image_id = int(self.temp_image.frame)
                else:
                    image_id = -1
                # print(pred_class + " detected at a distance of " + str(dist)+"!")
                for vehicle in vehicles:
                    # Check if vehicle is self.
                    true_class = vehicle.type_id

                    if vehicle.id == self._vehicle.id:
                        continue

                    ego_orientation = self._vehicle.get_transform().rotation.yaw
                    other_orientation = vehicle.get_transform().rotation.yaw
                    # if dist == 0:
                    #     print("Ego: {}".format(self._vehicle.get_transform().get_forward_vector()))
                    #     print("Other: {}".format(v._vehicle.get_transform().get_forward_vector()))
                    # Check if v is on same lane as self.
                    v_loc = vehicle.get_location()
                    # print("Distance from sensor: "+str(dist))
                    # print("CARLA computed distance: "+str(v_loc.distance(location)))
                    carla_dist = v_loc.distance(location)
                    if abs(dist - carla_dist+4.5) > 2:
                        image_id = int(self.temp_image.frame)
                        r_channel = self.depth_image[centre_x][centre_y][2]
                        g_channel = self.depth_image[centre_x][centre_y][1]
                        b_channel = self.depth_image[centre_x][centre_y][0]
                    else:
                        r_channel, g_channel, b_channel = "", "", ""
                    if abs(dist - carla_dist) <= 25:
                        false_decision_trigger = 0
                        if carla_dist < min_dist and dist >= min_dist:
                            false_decision_trigger = 1
                        self.row_values = [
                            str(dist),
                            str(carla_dist),
                            str(image_id),
                            str(x1),
                            str(y1),
                            str(x2),
                            str(y2),
                            str(r_channel),
                            str(g_channel),
                            str(b_channel),
                            str(false_decision_trigger),
                            pred_class,
                            true_class,
                            ego_orientation,
                            other_orientation
                            ]
                        # print("calculated distance is: {}".format(dist))

                    v_w = self._map.get_waypoint(v_loc)
                    if lane_id != v_w.lane_id:
                        continue

                    if forward and norm_dot(forward, v_loc - location) < 0.0:
                        continue

                    if dist < min_dist:

                        return (False, vehicle)

        return (True, None)



    def run_step(self):
        self_loc = self._vehicle.get_location()
        self_forward = self._vehicle.get_transform().get_forward_vector()
        is_clear, blocker = self.lane_clear(self_loc,
                                            min_dist=2.0*self.clear_dist,
                                            forward=self_forward)
        super().run_step() # goes to pid_agent.py line 76

        cur_w = self.waypoints[0]
        # print("self.waypoints: {}".format(cur_w.lane_change))
        if self.lane_state != 0: # if vehicle is not in its desired lane
            speed = self.target_speed * 1.5
            # print("We are increasing speed of ego")
            self.row_values.append("Increase")
        else:
            speed = self.target_speed
            self.row_values.append("Same")

        # print("Checking the is_clear variable")
        if not is_clear:
            self.row_values.append("Not clear")
            if not self.is_changing_lane:
                # print("Entering the code for changing lane")
                self.change_lane()

            # return self.controller.run_step(speed,
            #                                 self.waypoints[0], self.row_values,
            #                                 blocker.get_location(), self.calc_dist) # goes to pid_follow_controller.py line 52
            return self.controller.run_step(target_speed=speed, waypoint=self.waypoints[0], location=blocker.get_location())
        else:
            self.row_values.append("Clear")
            # print("Entered the code where is_clear is true")
            self.is_changing_lane = False
            if self.lane_state != 0:
                lane_change = LaneChange(-np.sign(self.lane_state))
                self.change_lane(lane_change=lane_change)

                return self.controller.run_step(speed, self.waypoints[0], blocker.get_location())

    def process_rgb_sensor_data(self, image:carla.libcarla.Image):
        """Listener function for camera"""
        img = np.array(image.raw_data)
        img = img.reshape((self.image_sizex, self.image_sizey, 4))
        img = img[:,:,:3]
        self.rgb_image = img
        self.temp_image = image

    def process_depth_sensor_data(self, image:carla.libcarla.Image):
        """Listener function for LiDAR/Depth"""
        self.fov = image.fov
        img = np.array(image.raw_data)
        img = img.reshape((self.image_sizex, self.image_sizey, 4))
        img = img[:,:,:3]
        self.depth_image = img
        self.temp_depth_image = image

    def get_detection(self, image, classes=None):
        # TODO :: Verify correct explaination
        """Get the first prediction from the model?"""
        image1 = letterbox(image, self.imgsz, stride=self.stride, auto=True)[0]
        image1 = image1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image1 = np.ascontiguousarray(image1)

        image1 = torch.from_numpy(image1).to(self.device1)
        image1 = image1.half() if self.od_model.fp16 else image1.float()  # uint8 to fp16/32
        image1 /= 255  # 0 - 255 to 0.0 - 1.0
        if len(image1.shape) == 3:
            image1 = image1[None]  # expand for batch dim

        pred = self.od_model(image1)

        # suppresses overlapping boxes and returns list of detections per image
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, self.agnostic_nms, max_det=self.max_det)
        for det in pred:
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(image1.shape[2:], det[:, :4], image.shape).round()
        pred = pred[0].tolist()
        return pred


    def compute_distance_from_depth_image(self, x: float, y: float) -> float:
        """Computes the distance based on the depth image"""
        dist = 1000*(self.depth_image[x][y][2] + (self.depth_image[x][y][1]*256) + (self.depth_image[x][y][0]*256*256))/(256*256*256-1)
        # if dist == 0:
        #     print("RGB values = {}".format(self.depth_image[x][y][:]))
        # print("dist is: {}".format(dist))
        # print("Value of x is: {}, and y is {}".format(x, y))
        # dataframe_array= pd.DataFrame(self.depth_image[x][y][:])
        # dataframe_array.to_csv("depthimage.csv")
        # print("dimensions are {}".format(self.depth_image.shape))
        fov = self.fov
        # print("fov is: {}".format(fov)) # field of view = 110
        fov = (fov/2 ) * math.pi / 180
        center = self.image_sizex / 2
        # print("center value is: {}".format(center)) # value is 540
        focal_length = (center)/(math.tan(fov))
        # print("focal length is: {}".format(focal_length)) # value is 378

        # print("Depth image is: {}".format(self.depth_image))
        # print("Dimension of depth image is: {}".format(self.depth_image.shape))

        new_x = ((x - center)*dist)/focal_length
        # print("old x was: {} and new x is: {}".format(x, new_x))
        new_y = ((y - center)*dist)/focal_length
        dist = math.sqrt(pow(new_x, 2) + pow(new_y, 2) + pow(dist, 2))
        # print("new calculated dist is: {}".format(dist))
        # if dist == 0:
        #     print("newx: {}, newy: {}".format(new_x, new_y))
        return dist
