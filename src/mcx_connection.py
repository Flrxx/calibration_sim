import motorcortex
import os
import logging.handlers
import json
import time
import copy

from csv_routines import read_dataset, write_dataset
from robot_control.motion_program import Waypoint, MotionProgram, PoseTransformer
from robot_control.system_defs import InterpreterStates, States, ModeCommands, StateEvents, Modes, InterpreterEvents, MotionGeneratorStates
from robot_control.robot_command import RobotCommand

import hayati_model
from math import sqrt, pi
import numpy as np
from math_routines import DEG

def get_parameters(path: str) -> None:
    try:
        with open(path) as f:
            templates = json.load(f)
        return templates
    except Exception as e:
        return e

class MCX:
    def __init__(self, config="config/ARM95.json"):
        self._path = self.__project_path()
        self.robot_parameters = get_parameters(
            os.path.join(self._path, "src/config", config))
        if not self.__connect("192.168.2.100"):
            raise Exception('Failed to connect') 


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
            self.subscription = self.sub.subscribe(
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
    
    def move_through_dataset(self, model: hayati_model.HayatiModel):
        dataset_poses = read_dataset(model.contribution_dataset, model.fieldnames_options["wire_contributions"])[:, 0:7]
        dataset_poses[:, 0:6] *= DEG
        self.motion_program.addMoveJ([Waypoint([0, 0, pi/2, 0, pi/2, 0])])
        self.execute_move()
        
        zero_pose = model.zero_wire_angles.copy()
        zero_point = self.robot_PoseTransformer.calcJointToCartPose(zero_pose).jointtocartlist[0].cartpose.coordinates

        self.motion_program.addMoveJ([Waypoint(zero_pose)])
        self.execute_move()

        result = np.zeros(shape=(len(dataset_poses), 8))
        for i, pose in enumerate(dataset_poses):
            result[i, 6] = pose[6]
            pose = pose[:6]

            tcp_coords_nom = self.robot_PoseTransformer.calcJointToCartPose(pose).jointtocartlist[0].cartpose.coordinates[0:3]
            distance_theor = sqrt((tcp_coords_nom[0] - zero_point[0])**2 + (tcp_coords_nom[1] - zero_point[1])**2 + (tcp_coords_nom[2] - zero_point[2])**2)
            
            vertical_offset =  copy.deepcopy(zero_point)
            vertical_offset[2] += distance_theor
            
            self.motion_program.addMoveL([Waypoint(vertical_offset)])
            self.execute_move()
            #time.sleep(1)
            pose_vertical_offset = self.joint_subscription.read()[0].value

            self.motion_program.addMoveJ([Waypoint(pose)])
            self.execute_move()
            #time.sleep(1)

            result[i, 7] = 99 # get wire dist
             # index
            result[i, 0:6] = self.joint_subscription.read()[0].value
            
            self.motion_program.addMoveJ([Waypoint(pose_vertical_offset)])
            self.execute_move()

            self.motion_program.addMoveL([Waypoint(vertical_offset)])
            self.execute_move()

            self.motion_program.addMoveJ([Waypoint(zero_pose)])
            self.execute_move()
            print(f"Pose {i + 1} done")
        return result
    
    def write_calibration_dataset(self, model: hayati_model.HayatiModel):
        calibration_dataset = self.move_through_dataset(model)
        calibration_dataset[:, 0:6] /= DEG
        write_dataset(calibration_dataset, model.calibration_dataset, model.fieldnames_options["wire_samples"])
        print("Done")
    
def main():
    with open("src/config/ARM95_calibration.json", 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    
    mcx = MCX(config="ARM95.json")
    mcx.write_calibration_dataset(model)

if __name__ == "__main__":
    main()
