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

    def get_closest_pose_to_euler_angle(self, x, y, z, position=None):
        search_angle = np.array([x, y, z])
        print(search_angle)

        distances = []
        for p in self.library:
            a = p['euler']
            # normalize the angle
            a = np.array([a['x'], a['y'], a['z']]) - np.array([0, 0, -1.57])
            distances.append(np.linalg.norm(a - search_angle))

        if position is not None:
            position = np.array(position)
            distances = np.array(distances) + np.array(
                [np.linalg.norm([a['pose'].position.x, a['pose'].position.y, a['pose'].position.z] - position) for a in self.library])

        min_dist = np.argsort(distances)

        if distances[min_dist[0]] < 0.001:
            print(self.library[min_dist[0]]['euler'])
            return self.library[min_dist[0]]['joints']
        else:
            print(self.library[min_dist[0]]['euler'])
            # weighted KNN 'predictor'
            num_neighbours = 3
            weights = np.array([distances[min_dist[i]] for i in range(num_neighbours)])
            weights = (1 / weights)
            normed_weights = weights / sum(weights)
            print(normed_weights)
            angles = [self.library[min_dist[i]]['joints'] for i in range(num_neighbours)]

            print('Min Dist 1', distances[min_dist[0]], self.library[min_dist[0]]['euler'])
            print('Min Dist 2', distances[min_dist[1]], self.library[min_dist[1]]['euler'])
            print('Min Dist 3', distances[min_dist[2]], self.library[min_dist[2]]['euler'])
            return self.get_average_joint_position(angles, normed_weights)

    def get_average_joint_position(self, joint_positions, weights):

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
        ], axis=0, weights=weights)

        ja.joint1 = avg[0]
        ja.joint2 = avg[1]
        ja.joint3 = avg[2]
        ja.joint4 = avg[3]
        ja.joint5 = avg[4]
        ja.joint6 = avg[5]
        ja.joint7 = avg[6]

        return ja

    def add_pose_to_library(self, pose, joints):
        print(pose)
        euler = euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        self.library.append({
            'pose': pose,
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
