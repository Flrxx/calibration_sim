import motorcortex
import os
import logging.handlers
import json
import time
import copy
import serial
import argparse


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
    def __init__(self, config="config/ARM95.json", is_real=True, start_num=2):
        self._path = self.__project_path()
        self.is_real = is_real
        self.start_num = start_num - 2
        #self.end_num = end_num - 2
        self.null_value = None
        self.robot_parameters = get_parameters(
            os.path.join(self._path, "src/config", config))
        if not self.__connect("192.168.2.100"):
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

    def execute_moveL(self, point: np.ndarray, velocity=0.02):
        self.motion_program.addMoveL([Waypoint(point)], velocity=velocity)
        self.execute_move()
    
    def execute_moveJ(self, pose: np.ndarray, rotational_velocity=0.1):
        self.motion_program.addMoveJ([Waypoint(pose)], rotational_velocity=rotational_velocity)
        self.execute_move()

    def find_next_positive_index(self, dataset_poses, start_num):
        for i in range(len(dataset_poses)):
            if dataset_poses[start_num + i, 6] >= 0:
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

    def move_through_dataset(self, dataset_poses: np.ndarray, zero_pose: np.ndarray, model):        
        zero_point = self.robot_PoseTransformer.calcJointToCartPose(zero_pose).jointtocartlist[0].cartpose.coordinates

        result = np.zeros(shape=(len(dataset_poses), 8))
        result[:, 6] = dataset_poses[:, 6]
        #dataset_poses = dataset_poses[:, :6]
        
        self.execute_moveJ(zero_pose)
        self.null_value = self.get_wire_distance(0)
        previous_poses = []
        last_index_extra_points = -1
        last_index = dataset_poses[0, 6]
        for i, pose in enumerate(dataset_poses): 
            if (last_index != pose[6] or last_index != self.find_next_positive_index(dataset_poses, i)) and dataset_poses[i-1, 6] >= 0 and pose[6] != -3:
                self.move_to_start(previous_poses, zero_point)

            if pose[6] < 0:
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

            elif dataset_poses[i - 1, 6] < 0 or pose[6] == last_index_extra_points:
                last_index_extra_points = pose[6]
                start_pose = self.joint_subscription.read()[0].value
                previous_poses.append([*start_pose, -1]) 

                print(f"Started pose {i + 2 + self.start_num}")     
                self.execute_moveJ(pose[:6])
                print(f"Done pose {i + 2 + self.start_num}")
                
                #time.sleep(2)
                tcp_coords_nom = self.robot_PoseTransformer.calcJointToCartPose(pose[:6]).jointtocartlist[0].cartpose.coordinates[0:3]
                distance_theor = sqrt((tcp_coords_nom[0] - zero_point[0])**2 + (tcp_coords_nom[1] - zero_point[1])**2 + (tcp_coords_nom[2] - zero_point[2])**2)
                print(f"Theoretical distance: {(distance_theor * 1000):.2f} mm")
                input()
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
                vertical_offset_point[2] += distance_theor
                print(f"Theoretical distance: {(distance_theor * 1000):.2f} mm")
            
                self.execute_moveL(vertical_offset_point)
                #time.sleep(1)
                vertical_offset_pose = self.joint_subscription.read()[0].value
                
                last_index = pose[6]   
                self.execute_moveJ(pose[:6])
                print(f"Done pose {i + 2 + self.start_num}")
                input()
                #time.sleep(2)
                wire_len = self.get_wire_distance(self.null_value)
                if wire_len > 0:
                    print(f"Wire lenght: {wire_len}")

                result[i, 7] = wire_len
                result[i, 0:6] = self.joint_subscription.read()[0].value
                self.execute_moveJ(vertical_offset_pose)
                #time.sleep(1)
            add_line_csv((result[i]), "datasets/ARM95/wire/test.csv", model.fieldnames_options["wire_samples"])
        self.move_to_start(previous_poses, zero_point)
        return result
    
    def write_calibration_dataset(self, model: hayati_model.HayatiModel):              
        contribution_dataset = read_dataset(model.contribution_dataset, model.fieldnames_options["wire_contributions"])[:, 0:7]
        contribution_dataset[:, 0:6] *= DEG
        contribution_dataset = contribution_dataset[self.start_num:, :]
        calibration_dataset = self.move_through_dataset(contribution_dataset, model.zero_wire_angles.copy(), model)
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
            #input()
            time.sleep(2)
            wire_len = self.get_wire_distance(self.null_value)
            if wire_len > 0:
                print(f"Wire lenght: {wire_len}")

            result[i, 0:6] = self.joint_subscription.read()[0].value
            result[i, 6] = np.linalg.norm(calculate_distance(model, result[i, 0:6], "nominal"))
            result[i, 7] = np.linalg.norm(calculate_distance(model, result[i, 0:6], "estimated"))
            result[i, 8] = wire_len / 1000
            result[i, 9] = abs(wire_len - result[i, 6]) - abs(wire_len - result[i, 7])
            add_line_csv((result[i]), "datasets/ARM95/wire/test.csv", model.fieldnames_options["test_result"])
            
            self.execute_moveJ(vertical_offset_pose)
            
            self.move_to_start([], zero_point)
        return result

    def write_validation_dataset(self, model: hayati_model.HayatiModel):              
        validation_dataset = read_dataset("datasets/ARM95/validation_dataset.csv", model.fieldnames_options["test_result"])
        validation_dataset[:, :6] *= DEG
        validation_dataset[:, 6:] /= 1000
        validation_dataset = validation_dataset[self.start_num:, :]
        validation_dataset = self.move_through_validation_dataset(validation_dataset, model.zero_wire_angles.copy(), model)
        
        validation_dataset[:, :6] /= DEG
        validation_dataset[:, 6:] *= 1000
        write_dataset(validation_dataset, "results/ARM95/validation_results.csv", model.fieldnames_options["test_result"])
        print("Done")
    
def main(args):
    with open("src/config/ARM95_calibration.json", 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    
    # is_real = False
    # start_num = 64
    mcx = MCX(config="ARM95.json", is_real=args.is_real, start_num=args.start_num)
    if args.mode == "calibration":
        mcx.write_calibration_dataset(model)
    elif args.mode == "validation":
        mcx.write_validation_dataset(model)
    else: 
        print("Wrong mode")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="calibration or validation", default="calibration")
    parser.add_argument("-s", "--start_num", help="Choose model to be visualized. Default: nominal", type=int, default=2)
    parser.add_argument("--is_real", action='store_true')
    args = parser.parse_args()
    main(args)