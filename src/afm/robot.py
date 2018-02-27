import sys

import actionlib
import rospy
import numpy as np
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from kinova_msgs.msg import JointAngles, ArmJointAnglesGoal, ArmJointAnglesAction
from afm.camera import CameraThread
import moveit_commander
import pickle
import os
from afm.data import RobotData


# import moveit_msgs.msg
# import geometry_msgs.msg

def norm_q(q):
    a, b, c, d = q
    Z = np.sqrt(a ** 2 + b ** 2 + c ** 2 + d ** 2)
    return np.array([a / Z, b / Z, c / Z, d / Z])


class RobotHandler:

    def __init__(self):

        print("============ started robot init")

        self.camera = None
        self.REAL_ROBOT_CONNECTED = False
        self.robot_pose = PoseStamped()
        self.robot_joint_state = JointState()
        self.robot_joint_angles = JointAngles()
        self.robot_joint_command = JointAngles()

        # currently required minimum initialization
        rospy.init_node('afm', anonymous=True)
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.group = moveit_commander.MoveGroupCommander("arm")
        self.group.set_goal_orientation_tolerance(0.0001)

        print("============ robot init successful")

        pass

    def spin(self):
        rospy.spin()

    def receive_pose_data(self, robot_pose):
        self.robot_pose = robot_pose

    def receive_joint_state(self, robot_joint_state):
        self.robot_joint_state = robot_joint_state

    def receive_joint_command(self, robot_joint_angles):
        self.robot_joint_angles = robot_joint_angles

    def receive_joint_angles(self, robot_joint_command):
        self.robot_joint_command = robot_joint_command

    def set_camera_flag(self, state):
        self.camera.FLAG = state

    def get_current_euler(self):
        real_q = self.robot_pose.pose.orientation
        real_euler = euler_from_quaternion(np.array([real_q.x, real_q.y, real_q.z, real_q.w]))

        # do some remapping
        real_euler = np.array([real_euler[1], real_euler[0] * -1, real_euler[2] + 1.57])
        return real_euler

    def get_difference(self, planned_q, planned_coord):
        # temp
        # print(planned_coord, planned_q)

        real_euler = self.get_current_euler()
        real_position = self.robot_pose.pose.position

        difference_position = np.array([real_position.x, real_position.y, real_position.z]) - np.array(planned_coord)

        print(difference_position)

        # EULER
        planned_euler = np.array(euler_from_quaternion(planned_q))
        difference_euler = np.array(real_euler - planned_euler)
        print(difference_euler)
        # collect difference in qaternion as well

        return difference_position, difference_euler  # , difference_quanternion

    def rotate_arm(self, angles, position):

        print("============ Rotating arm")

        pose_target = Pose()

        for a in angles:

            q = quaternion_from_euler(*a)

            pose_target.orientation.x = q[0]
            pose_target.orientation.y = q[1]
            pose_target.orientation.z = q[2]
            pose_target.orientation.w = q[3]
            pose_target.position.x = position[0]
            pose_target.position.y = position[1]
            pose_target.position.z = position[2]

            self.group.set_pose_target(pose_target)

            # plan1 = self.group.plan()

            if self.camera is not None and self.camera.has_slid:
                print("SLIDING RECEIVED, STOPPING")
                # self.camera.has_slid = False
                # self.group.execute(plan1)
                return "SLIDING"

            print("Going to " + str(max(a) * (180 / np.pi)) + " degree")

            # run command async so camera can collect data

            self.group.go(wait=False)
            rospy.sleep(1)

            if self.REAL_ROBOT_CONNECTED:
                self.get_difference(q, position)

            # RESET
            self.group.clear_pose_targets()

            if rospy.is_shutdown():
                exit(0)

        return "DONE"

    def reset_arm(self):

        q = quaternion_from_euler(0, 0, 0)
        position = [0, 0.6, 0.5]

        self.set_arm_position(position, q)

    def set_arm_position(self, position, orientation):

        pose_target = Pose()

        pose_target.orientation.x = orientation[0]
        pose_target.orientation.y = orientation[1]
        pose_target.orientation.z = orientation[2]
        pose_target.orientation.w = orientation[3]
        pose_target.position.x = position[0]
        pose_target.position.y = position[1]
        pose_target.position.z = position[2]

        self.group.set_pose_target(pose_target)

        self.group.go(wait=True)

        self.group.clear_pose_targets()

        return pose_target

    def run_debug_code(self):

        test_pose_1 = Pose()
        test_pose_1.orientation.x = 0.01
        test_pose_2 = Pose()
        test_pose_2.orientation.x = 0.01
        print(test_pose_1 == test_pose_2)

    def rerun_calibration(self):
        rd = RobotData()

        dirpath = rd.base_path + '_rerun'
        os.mkdir(dirpath)

        print("FOUND " + str(len(rd.files)))
        for i in range(len(rd.files)):
            joint_positions = rd.get_average_joint_position(i)
            print(joint_positions)
            # rospy.sleep(1)
            self.joint_angle_client(joint_positions)
            # todo add proper tracking at some point
            # self.collect_pose_data(dirpath)

    def joint_angle_client(self, angle_set):
        """Send a joint angle goal to the action server."""
        action_address = '/j2n6s300_driver/joints_action/joint_angles'
        client = actionlib.SimpleActionClient(action_address,
                                              ArmJointAnglesAction)
        client.wait_for_server()

        goal = ArmJointAnglesGoal()

        goal.angles.joint1 = angle_set['joint1']
        goal.angles.joint2 = angle_set['joint2']
        goal.angles.joint3 = angle_set['joint3']
        goal.angles.joint4 = angle_set['joint4']
        goal.angles.joint5 = angle_set['joint5']
        goal.angles.joint6 = angle_set['joint6']
        goal.angles.joint7 = angle_set['joint7']

        client.send_goal(goal)
        if client.wait_for_result(rospy.Duration(20)):
            rospy.sleep(3.0)
            return client.get_result()
        else:
            print('        the joint angle action timed-out')
            client.cancel_all_goals()

        return None

    def collect_pose_data(self, dirpath):
        a = ()
        collected_data = {
            'angle': a,
            'robot_pose': [PoseStamped()],
            'robot_joint_state': [],
            'robot_joint_angles': [],
            'robot_joint_command': [],
            'planned_pose': None,
            'time': [rospy.get_time(), rospy.get_time()]
        }

        print('Collecting Data')
        collected_data['time'][0] = rospy.get_time()
        while not rospy.is_shutdown() and len(collected_data['robot_pose']) < 51:
            rospy.sleep(0.05)
            if collected_data['robot_pose'][-1].header.seq != self.robot_pose.header.seq:
                collected_data['robot_pose'].append(self.robot_pose)
                collected_data['robot_joint_state'].append(self.robot_joint_state)
                collected_data['robot_joint_angles'].append(self.robot_joint_angles)
                collected_data['robot_joint_command'].append(self.robot_joint_command)

        with open(dirpath + '/' + str(a) + '.pickle', 'wb') as f:
            print("Got " + str(len(collected_data['robot_pose'])))
            # remove initial (empty) state
            collected_data['robot_pose'] = collected_data['robot_pose'][1:]
            # get finished time
            collected_data['time'][1] = rospy.get_time()
            # save data
            pickle.dump(collected_data, f)
            print('Finished Data Collection')


    def run_calibration(self):
        dirpath = '/home/keno/data/' + str(rospy.get_time())
        os.mkdir(dirpath)

        positions = [[0, 0.6, 0.5], [0, 0.6, 0.5], [0, 0.3, 0.7], [0, 0.5, 0.7]]

        all_angles = [
            [(0, i * np.pi, 0) for i in np.linspace(0, 0.25, 90)],
            [(0, - 1 * i * np.pi, 0) for i in np.linspace(0, 0.5, 90)],
            [(i * np.pi, 0, 0) for i in np.linspace(0, 0.375, 90)],
            [(-1 * i * np.pi, 0, 0) for i in np.linspace(0, 0.375, 90)]
        ]

        for i in range(4):

            position = positions[i]
            angles = all_angles[i]

            for a in angles:

                collected_data = {
                    'angle': a,
                    'robot_pose': [PoseStamped()],
                    'robot_joint_state': [],
                    'robot_joint_angles': [],
                    'robot_joint_command': [],
                    'planned_pose': None,
                    'time': [rospy.get_time(), rospy.get_time()]
                }

                q = quaternion_from_euler(*a)

                print("Going to " + str(max(np.abs(a)) * (180 / np.pi)) + " degree")
                collected_data['planned_pose'] = self.set_arm_position(position, q)

                # wait for systems to catch up
                rospy.sleep(1)

                # collect data
                print('Collecting Data')
                collected_data['time'][0] = rospy.get_time()
                while not rospy.is_shutdown() and len(collected_data['robot_pose']) < 51:
                    rospy.sleep(0.05)
                    if collected_data['robot_pose'][-1].header.seq != self.robot_pose.header.seq:
                        collected_data['robot_pose'].append(self.robot_pose)
                        collected_data['robot_joint_state'].append(self.robot_joint_state)
                        collected_data['robot_joint_angles'].append(self.robot_joint_angles)
                        collected_data['robot_joint_command'].append(self.robot_joint_command)

                # if self.REAL_ROBOT_CONNECTED:
                #     self.get_difference(q, position)

                with open(dirpath + '/' + str(a) + '.pickle', 'wb') as f:
                    print("Got " + str(len(collected_data['robot_pose'])))
                    # remove initial (empty) state
                    collected_data['robot_pose'] = collected_data['robot_pose'][1:]
                    # get finished time
                    collected_data['time'][1] = rospy.get_time()
                    # save data
                    pickle.dump(collected_data, f)
                    print('Finished Data Collection')

                if rospy.is_shutdown():
                    exit(0)

            break

            for a in reversed(angles[:-1]):
                q = quaternion_from_euler(*a)

                print("Resetting to " + str(max(np.abs(a)) * (180 / np.pi)) + " degree")
                self.set_arm_position(position, q)

                rospy.sleep(1)

            # go to position x

            # record position from robot

            # record joint positions

            # do it in 5 degree steps for + / - 90 degrees in both directions

    def connect_to_camera(self):

        topics = [name for (name, _) in rospy.get_published_topics()]

        if '/raspicam_node/image/compressed' in topics:
            print("============ FOUND camera")
            print("============ Subscribing to /raspicam_node/image/compressed")
            self.camera = CameraThread()
            self.camera.start()
            self.set_camera_flag('IGNORE')
        else:
            print("============ COULD NOT find camera, running blind")

    def connect_to_real_robot(self):

        topics = [name for (name, _) in rospy.get_published_topics()]

        if '/j2n6s300_driver/out/tool_pose' in topics:
            print("============ FOUND real robot")
            print("============ Subscribing to topics")
            rospy.Subscriber("/j2n6s300_driver/out/tool_pose", PoseStamped, self.receive_pose_data)
            rospy.Subscriber("/j2n6s300_driver/out/joint_state", JointState, self.receive_joint_state)
            rospy.Subscriber("/j2n6s300_driver/out/joint_angles", JointAngles, self.receive_joint_angles)
            rospy.Subscriber("/j2n6s300_driver/out/joint_command", JointAngles, self.receive_joint_command)
            self.REAL_ROBOT_CONNECTED = True
        else:
            print("============ COULD NOT find real robot")

    def run_demo_experiment(self):

        print("============ Resetting arm")

        self.reset_arm()

        angles = [(0, i * np.pi, 0) for i in np.linspace(0, 0.5, 50)]

        print("============ Running test")
        position = [0, 0.6, 0.5]
        status = self.rotate_arm(angles, position)

        if status == 'SLIDING':
            # reset to 0
            print(self.get_current_euler())
            self.reset_arm()

        if status == 'DONE':
            # reset to 0
            self.reset_arm()

        self.shutdown()

    def calibrate_camera(self):
        if self.camera is None:
            print("Skipping Calibration, no camera detected")
            return

        self.camera.start_calibration()

        while not rospy.is_shutdown() and self.camera.FLAG == 'CALIBRATE':
            rospy.sleep(1)
            pass

    def shutdown(self):

        print("============ WAITING ON CAMERA FOR SHUTDOWN")
        # self.set_camera_flag('SHUTDOWN')
        if self.camera is not None:
            self.camera.join()
        print("============ SHUTDOWN COMPLETED")
        print("============ Everything finished. Now Crashing. RIP")

        # it fails because its a know issue
        # https://github.com/ros-planning/moveit/issues/331
        moveit_commander.roscpp_shutdown()
        return exit(1)
