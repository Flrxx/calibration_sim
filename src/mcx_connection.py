import motorcortex
import os
import logging.handlers
import json
import time
import copy
import serial
import argparse
from scipy.spatial.transform import Rotation

from dataset_generation_wire import calculate_distance
from csv_routines import read_dataset, write_dataset, add_line_csv
from robot_control.motion_program import Waypoint, MotionProgram, PoseTransformer
from robot_control.system_defs import InterpreterStates, States, ModeCommands, StateEvents, Modes, InterpreterEvents, MotionGeneratorStates
from robot_control.robot_command import RobotCommand

import hayati_model
from math import sqrt, pi
import numpy as np
from math_routines import DEG

PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600

def get_parameters(path: str) -> None:
    try:
        with open(path) as f:
            templates = json.load(f)
        return templates
    except Exception as e:
        return e

class MCX:
    def __init__(self, ip, config="config/ARM95.json", is_real=True, start_num=2):
        self._path = self.__project_path()
        self.is_real = is_real
        self.start_num = start_num - 2
        #self.end_num = end_num - 2
        self.null_value = None
        self.robot_parameters = get_parameters(
            os.path.join(self._path, "src/config", config))
        if not self.__connect(ip):
            raise Exception('Failed to connect') 
        try:
            if self.is_real:
                self.ser = serial.Serial(PORT, BAUD_RATE, timeout=2)
                print(f"--- Connected to {PORT} ---")
        except serial.SerialException as e:
            print(f"Error: {e}")


    def __project_path(self) -> str:
        current_path = os.path.abspath(__file__)
        current_path = current_path.split("/")
        current_path = current_path[:-2]
        current_path = "/".join(current_path)
        return current_path
        
    def __connect(self, ip: str):
        parameter_tree = motorcortex.ParameterTree()
        self.motorcortex_types = motorcortex.MessageTypes()
        license_file = os.path.join(
            self._path, 'src', 'license', 'mcx.cert.pem')
        
        try:            
            self.req, self.sub = motorcortex.connect(f'wss://{ip}:5568:5567', self.motorcortex_types, parameter_tree,
                                                    timeout_ms=1000, certificate=license_file,
                                                    login="admin", password="motioncore")
            self.pose_subscription = self.sub.subscribe(
                [self.robot_parameters['robot']['paths']['toolCoordinate']], 'group1', 1)
            self.joint_subscription = self.sub.subscribe(
                [self.robot_parameters['robot']['paths']['jointPosition']], 'group2', 1)
        

        except Exception as e:
            logging.error(f"ip: {ip}")
            logging.error(f"license path: {license_file}")
            logging.error(f"Failed to establish connection: {e}")
            return False

        self.robot = RobotCommand(self.req, self.motorcortex_types)

        if self.robot.engage():
            logging.info('Robot is at Engage')
        else:
            logging.error('Failed to set robot to Engage')
            return False
        
        self.motion_program = MotionProgram(self.req, self.motorcortex_types)
        self.robot_PoseTransformer = PoseTransformer(self.req, self.motorcortex_types)

        self.robot.reset()
        return True
    
    def execute_move(self):
        self.motion_program.send("program").get()
        if self.robot.play() is InterpreterStates.MOTION_NOT_ALLOWED_S.value:
            logging.info('Robot is not at a start position, moving to the start')
            if self.robot.moveToStart(200):
                logging.info('Robot is at the start position')
            else:
                raise Exception('Failed to move to the start position')
        self.robot.play(0)
        while self.robot.getState() is InterpreterStates.PROGRAM_RUN_S.value:
            time.sleep(0.1)
        self.motion_program.clear()

    def get_wire_distance(self, null_value):
        if self.is_real:
            self.ser.reset_input_buffer() 
            line = self.ser.readline().decode('utf-8').strip()
            value = float(line) # mm
            value -= null_value
        else:
            value = -1
        return value

    def execute_moveL(self, point: np.ndarray, velocity=0.1): #0.03
        self.motion_program.addMoveL([Waypoint(point)], velocity=velocity)
        self.execute_move()
    
    def execute_moveJ(self, pose: np.ndarray, rotational_velocity=0.5): #0.1
        self.motion_program.addMoveJ([Waypoint(pose)], rotational_velocity=rotational_velocity)
        self.execute_move()

    def find_next_positive_index(self, dataset_poses, start_num):
        for i in range(len(dataset_poses)):
            #if dataset_poses[start_num + i, 6] >= 0:
            if isinstance(dataset_poses[start_num + i, 6], str):
                return dataset_poses[start_num + i, 6]
        return -1

    def move_to_start(self, previous_poses, zero_point):
        print("Come to start")
        if len(previous_poses) == 0:
            self.execute_moveL(zero_point)
        else:
            for _ in range(len(previous_poses)):
                prev_pose = previous_poses.pop()
                if prev_pose[-1] == -1:
                    self.execute_moveJ(prev_pose[:6])
                elif prev_pose[-1] == -2:
                    self.execute_moveL(prev_pose[:6])
            previous_poses.clear()
        time.sleep(2)
        self.null_value = self.get_wire_distance(0)
        if self.null_value != -1:
            print(f"Null value: {self.null_value}")

    def move_through_cal_dataset(self, dataset_poses: np.ndarray, zero_pose: np.ndarray, model):        
        zero_point = self.robot_PoseTransformer.calcJointToCartPose(zero_pose).jointtocartlist[0].cartpose.coordinates

        result = np.array(dataset_poses[:, :8], dtype=object)
        
        self.execute_moveJ(zero_pose)
        self.null_value = self.get_wire_distance(0)
        previous_poses = []
        last_index_extra_points = -1
        last_index = dataset_poses[0, 6]
        for i, pose in enumerate(dataset_poses): 
            if (last_index != pose[6] or last_index != self.find_next_positive_index(dataset_poses, i)) and isinstance(dataset_poses[i-1, 6], str) and pose[6] != -3:
                self.move_to_start(previous_poses, zero_point)

            if isinstance(pose[6], int):
                print(f"Extra pose {i + 2 + self.start_num}")
                if pose[6] == -1 or pose[6] == -3:
                    start_pose = self.joint_subscription.read()[0].value
                    previous_poses.append([*start_pose, -1]) 
                    self.execute_moveJ(pose[:6])
                    result[i, 7] = -1
                    result[i, 0:6] = self.joint_subscription.read()[0].value
                   
                elif pose[6] == -2:
                    start_coords = self.pose_subscription.read()[0].value
                    previous_poses.append([*start_coords, -2]) 
                    pose_coords = self.robot_PoseTransformer.calcJointToCartPose(pose[:6]).jointtocartlist[0].cartpose.coordinates
                    self.execute_moveL(pose_coords)
                    result[i, 7] = -1
                    result[i, 0:6] = self.joint_subscription.read()[0].value
                else:
                    raise IndexError("Wrong index")
                result[i, 7] = -1
                result[i, 0:6] = self.joint_subscription.read()[0].value

            elif isinstance(dataset_poses[i - 1, 6], int) or pose[6] == last_index_extra_points:
                last_index_extra_points = pose[6]
                start_pose = self.joint_subscription.read()[0].value
                previous_poses.append([*start_pose, -1]) 

                print(f"Started pose {i + 2 + self.start_num}")     
                self.execute_moveJ(pose[:6])
                print(f"Done pose {i + 2 + self.start_num}")
                
                
                tcp_coords_nom = self.robot_PoseTransformer.calcJointToCartPose(pose[:6]).jointtocartlist[0].cartpose.coordinates[0:3]
                distance_theor = sqrt((tcp_coords_nom[0] - zero_point[0])**2 + (tcp_coords_nom[1] - zero_point[1])**2 + (tcp_coords_nom[2] - zero_point[2])**2)
                print(f"Theoretical distance: {(distance_theor * 1000):.2f} mm")
                time.sleep(3)
                #input()
                wire_len = self.get_wire_distance(self.null_value)
                if wire_len > 0:
                    print(f"Wire lenght: {wire_len}")
                
                
                #time.sleep(1)
                result[i, 7] = wire_len
                result[i, 0:6] = self.joint_subscription.read()[0].value
                last_index = pose[6]
            else:            
                print(f"Started pose {i + 2 + self.start_num}")     
                tcp_coords_nom = self.robot_PoseTransformer.calcJointToCartPose(pose[:6]).jointtocartlist[0].cartpose.coordinates[0:3]
                distance_theor = sqrt((tcp_coords_nom[0] - zero_point[0])**2 + (tcp_coords_nom[1] - zero_point[1])**2 + (tcp_coords_nom[2] - zero_point[2])**2)
                
                vertical_offset_point =  copy.deepcopy(zero_point)
                vertical_offset_point[2] += distance_theor + 50/1000 #mm
                print(f"Theoretical distance: {(distance_theor * 1000):.2f} mm")
            
                self.execute_moveL(vertical_offset_point)
                #time.sleep(1)
                vertical_offset_pose = self.joint_subscription.read()[0].value
                
                last_index = pose[6]   
                self.execute_moveJ(pose[:6])
                print(f"Done pose {i + 2 + self.start_num}")
                #input()
                time.sleep(3)
                wire_len = self.get_wire_distance(self.null_value)
                if wire_len > 0:
                    print(f"Wire lenght: {wire_len}")

                result[i, 7] = wire_len
                result[i, 0:6] = self.joint_subscription.read()[0].value
                self.execute_moveJ(vertical_offset_pose)
                
                #time.sleep(1) 
            add_line_csv((np.concatenate( (result[i, 0:6] / DEG, result[i, 6:]) )), "datasets/ARM95/wire/log.csv", model.fieldnames_options["wire_samples"])
        self.move_to_start(previous_poses, zero_point)
        return result
    
    def write_calibration_dataset(self, model: hayati_model.HayatiModel):              
        contribution_dataset = read_dataset(model.contribution_dataset, model.fieldnames_options["wire_contributions"])[:, 0:8]
        contribution_dataset[:, 0:6] *= DEG
        contribution_dataset = contribution_dataset[self.start_num:, :]
        calibration_dataset = self.move_through_cal_dataset(contribution_dataset, model.zero_wire_angles.copy(), model)
        calibration_dataset[:, 0:6] /= DEG
        write_dataset(calibration_dataset, model.calibration_dataset, model.fieldnames_options["wire_samples"])
        print("Done")

    def move_through_validation_dataset(self, dataset_poses: np.ndarray, zero_pose: np.ndarray, model):
        with open(model.results_file, 'r') as config_file:
            config = json.load(config_file)
        model.estimated_dh = copy.deepcopy(config["estimated_dh"])
        
        zero_point = self.robot_PoseTransformer.calcJointToCartPose(zero_pose).jointtocartlist[0].cartpose.coordinates

        result = np.zeros(shape=(len(dataset_poses), len(model.fieldnames_options["test_result"])))
        #dataset_poses = dataset_poses[:, :6]
        
        self.execute_moveJ(zero_pose)
        self.null_value = self.get_wire_distance(0)
        for i, pose in enumerate(dataset_poses): 
            print(f"Started pose {i + 2 + self.start_num}")     
            tcp_coords_nom = self.robot_PoseTransformer.calcJointToCartPose(pose[:6]).jointtocartlist[0].cartpose.coordinates[0:3]
            distance_theor = sqrt((tcp_coords_nom[0] - zero_point[0])**2 + (tcp_coords_nom[1] - zero_point[1])**2 + (tcp_coords_nom[2] - zero_point[2])**2)
            
            vertical_offset_point =  copy.deepcopy(zero_point)
            vertical_offset_point[2] += distance_theor
            print(f"Theoretical distance: {(distance_theor * 1000):.2f} mm")
        
            self.execute_moveL(vertical_offset_point)
            #time.sleep(1)
            vertical_offset_pose = self.joint_subscription.read()[0].value
            
            self.execute_moveJ(pose[:6])
            print(f"Done pose {i + 2 + self.start_num}")
            time.sleep(2)
            wire_len = self.get_wire_distance(self.null_value)
            if wire_len > 0:
                print(f"Wire lenght: {wire_len}")

            result[i, 0:6] = self.joint_subscription.read()[0].value
            result[i, 6] = np.linalg.norm(calculate_distance(model, result[i, 0:6], "nominal"))
            result[i, 7] = np.linalg.norm(calculate_distance(model, result[i, 0:6], "estimated"))
            result[i, 8] = wire_len / 1000
            result[i, 9] = abs(wire_len - result[i, 6]) - abs(wire_len - result[i, 7])
            add_line_csv((result[i]), "datasets/ARM95/wire/log.csv", model.fieldnames_options["test_result"])
            
            self.execute_moveJ(vertical_offset_pose)
            
            self.move_to_start([], zero_point)
        return result

    def write_validation_dataset(self, model: hayati_model.HayatiModel):              
        validation_dataset = read_dataset("datasets/ARM95/wire/validation_dataset.csv", model.fieldnames_options["test_result"])
        validation_dataset[:, :6] *= DEG
        validation_dataset[:, 6:] /= 1000
        validation_dataset = validation_dataset[self.start_num:, :]
        validation_dataset = self.move_through_validation_dataset(validation_dataset, model.zero_wire_angles.copy(), model)
        
        validation_dataset[:, :6] /= DEG
        validation_dataset[:, 6:] *= 1000
        write_dataset(validation_dataset, "results/ARM95/validation_results.csv", model.fieldnames_options["test_result"])
        print("Done")

    def calculate_wire_angles(self, model: hayati_model.HayatiModel, pose: np.ndarray):
        point = self.robot_PoseTransformer.calcJointToCartPose(pose).jointtocartlist[0].cartpose.coordinates
        point_coords = point[0:3]
        euler_angles = point[3:]
        wire_vec = point_coords - model.zero_offset_nominal
        wire_vec /= np.linalg.norm(wire_vec)

        #point_coords /= np.linalg.norm(point_coords)
        
        prod_dir = np.dot(model.zero_wire_direction, wire_vec)
        angle_dir = np.arccos(prod_dir) / DEG

        r = Rotation.from_euler('zyx', euler_angles)
        z_dir = r.as_matrix()[0:3, 2]
        prod_z = np.dot(z_dir, -wire_vec)
        angle_z = np.arccos(prod_z) / DEG
        return angle_dir, angle_z

    def move_through_angle_dataset(self, model: hayati_model.HayatiModel, dataset_poses: np.ndarray):
        zero_pose = model.zero_wire_angles
        zero_point = self.robot_PoseTransformer.calcJointToCartPose(zero_pose).jointtocartlist[0].cartpose.coordinates

        result = np.zeros(shape=(dataset_poses.shape[0], dataset_poses.shape[1] + 3))
        #dataset_poses = dataset_poses[:, :6]
        
        self.execute_moveJ(zero_pose)
        self.null_value = self.get_wire_distance(0)
        start_pose = self.robot_PoseTransformer.calcJointToCartPose(dataset_poses[0]).jointtocartlist[0].cartpose.coordinates
        for i, pose in enumerate(dataset_poses): 
            print(f"Started pose {i + 2 + self.start_num}")     
            tcp_coords_nom = self.robot_PoseTransformer.calcJointToCartPose(pose).jointtocartlist[0].cartpose.coordinates[0:3]
            distance_theor = sqrt((tcp_coords_nom[0] - zero_point[0])**2 + (tcp_coords_nom[1] - zero_point[1])**2 + (tcp_coords_nom[2] - zero_point[2])**2)
            
            # vertical_offset_point =  copy.deepcopy(zero_point)
            # vertical_offset_point[2] += distance_theor
            vertical_offset_point = start_pose
            print(f"Theoretical distance: {(distance_theor * 1000):.2f} mm")
        
            self.execute_moveL(vertical_offset_point)
            #time.sleep(1)
            vertical_offset_pose = self.joint_subscription.read()[0].value
            
            self.execute_moveJ(pose)
            print(f"Done pose {i + 2 + self.start_num}")
            time.sleep(2)
            wire_len = self.get_wire_distance(self.null_value)
            if wire_len > 0:
                print(f"Wire lenght: {wire_len}")

            result[i, 0:6] = self.joint_subscription.read()[0].value
            result[i, 6] = wire_len / 1000 - distance_theor
            angle_dir, angle_z = self.calculate_wire_angles(model, result[i, 0:6])
            print(angle_dir, angle_z)

            result[i, 7] = angle_dir
            result[i, 8] = angle_z
            
            add_line_csv((result[i]), "datasets/ARM95/wire/log.csv", ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'd_real', 'ang_dir', 'angle_z'])
            self.execute_moveJ(vertical_offset_pose)
            self.move_to_start([], zero_point)
        return result

    def write_angle_error(self, model: hayati_model.HayatiModel, poses: np.ndarray):
        res = self.move_through_angle_dataset(model, poses)
        res[:, :6] /= DEG
        res[:, 6] *= 1000
        write_dataset(res, "results/ARM95/angle_test_results.csv", ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'd_real', 'ang_dir', 'angle_z'], tolerance="0.2f")

    
def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    
    # is_real = False
    # start_num = 64
    mcx = MCX(args.ip, config="ARM95.json", is_real=args.is_real, start_num=args.start_num)
    if args.mode == "calibration":
        mcx.write_calibration_dataset(model)
    elif args.mode == "validation":
        mcx.write_validation_dataset(model)
    elif args.mode == "angle_error":
        poses = np.array([
                            [14.65, -1.97, 129.8, -37.84, 89.99, -14.64],
                            
                            [-38.13, 38.2, 102.69, -50.89, 89.99, 38.14],
                            [-38.13, 25.09, 94.9, -30, 89.99, 38.14],
                            [-23.66,  11.98, 114.02, -36, 89.99, 23.67],
                            [-9.3, 4.06, 123.61, -37.67, 89.99, 9.31],
                            [-0.85, 0.97, 126.9, -37.88, 89.99, 0.87],
                            
                            [26.36, -1.64, 129.49, -37.85, 89.99, -26.35],
                            [39.02,  1.9, 125.93, -37.83, 89.99, -39.01],
                            [49.88, 8.79, 118.07, -36.86, 89.99, -49.87], 
                            [59.67, 20.49, 102.01, -32.51, 89.99, -59.66], 
                            [59.67, 34.85, 110.02, -54.87, 89.99, -59.66],                             
                            ])
        poses *= DEG
        mcx.write_angle_error(model, poses)
    else: 
        print("Wrong mode")
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="", default="src/config/ARM95_calibration.json")
    parser.add_argument("--ip", help="", default="192.168.2.100")
    parser.add_argument("-m", "--mode", help="calibration or validation", default="angle_error")
    parser.add_argument("-s", "--start_num", help="Choose model to be visualized. Default: nominal", type=int, default=2)
    parser.add_argument("--is_real", action='store_true')
    args = parser.parse_args()
    main(args)