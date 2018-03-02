import numpy as np
import pickle
from geometry_msgs.msg import Pose
from kinova_msgs.msg import JointAngles
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import os
import rospy
class PoseLibrary:

    def __init__(self, path):
        self.library = [{'pose': Pose(), 'joints': JointAngles(), 'euler': {}, 'time': rospy.get_time()}]
        self.state = 'NONE'
        self.path = path
        self.load_library()
        pass

    def get_closest_pose_to_euler_angle(self, x, y, z):
        search_angle = np.array([x, y, z])
        # redo
        distances = [np.linalg.norm(np.array([a['x'], a['y'], a['z']]) - search_angle) for _, _, a in self.library]
        min_dist_index = np.argmin(distances)
        return self.library[min_dist_index]

    def add_pose_to_library(self, pose, joints):
        print(pose)
        euler = euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        self.library.append({
            'pose':pose,
            'joints': joints,
            'euler': {'x': euler[0], 'y': euler[1], 'z': euler[2]},
            'time': rospy.get_time()
        })
        self.save_library()

    def save_library(self):
        old_state = self.state
        self.state = 'SAVING'
        with open(self.path, 'wb') as f:
            pickle.dump(self.library, f)

        self.state = old_state
        pass

    def print_pose_library(self):
        print(self.library)
        pass

    def print_pose_index(self, index):
        print(self.library[index])

    def load_library(self):
        self.state = 'LOADING'
        if os.path.isfile(self.path):
            with open(self.path, 'rb') as f:
                self.library = pickle.load(f)
        else:
            self.library = []
        self.state = 'READY'
        pass
