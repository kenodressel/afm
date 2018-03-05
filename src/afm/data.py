import pickle
import numpy as np
import glob
import os
from kinova_msgs.msg import JointAngles


class RobotData:

    def __init__(self, base_path=None):
        self.base_path = base_path
        if base_path is None:
            dirs = sorted([x[0] for x in os.walk('/home/keno/data/')], key=os.path.getmtime)
            print(dirs)
            self.base_path = dirs[-2]
        self.files = []
        self.load_options()

        pass

    def load_options(self):
        self.files = sorted(glob.glob(self.base_path + '/*.pickle'), key=os.path.getmtime)
        pass

    def load_file_index(self, index):
        with open(self.files[index], 'rb') as f:
            data = pickle.load(f)
            return data

    def get_average_joint_position(self, joint_positions):

        ja = JointAngles()

        avg = np.average([
            [
                d.joint1,
                d.joint2,
                d.joint3,
                d.joint4,
                d.joint5,
                d.joint6,
                d.joint7
            ] for d in joint_positions
        ], axis=0)

        ja.joint1 = avg[0]
        ja.joint2 = avg[1]
        ja.joint3 = avg[2]
        ja.joint4 = avg[3]
        ja.joint5 = avg[4]
        ja.joint6 = avg[5]
        ja.joint7 = avg[6]

        return ja