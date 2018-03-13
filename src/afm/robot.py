import glob
import sys

import actionlib
import rospy
import numpy as np
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState, Imu
from kinova_msgs.msg import JointAngles, ArmJointAnglesGoal, ArmJointAnglesAction
from afm.camera import CameraThread
import moveit_commander
import pickle
import os
from afm.data import RobotData


def norm_q(q):
    a, b, c, d = q
    Z = np.sqrt(a ** 2 + b ** 2 + c ** 2 + d ** 2)
    return np.array([a / Z, b / Z, c / Z, d / Z])


class RobotHandler:
    """
    This is the main class of the robot controller
    It contains all methods and takes care of the camera handling (via a thread)
    """

    def __init__(self):

        # currently required minimum initialization
        rospy.init_node('afm', anonymous=True)

        rospy.loginfo("Started Robot Init")

        # Status variables
        self.REAL_ROBOT_CONNECTED = False
        self.IMU_CONNTECTED = False

        # Callback Data holders
        self.robot_pose = PoseStamped()
        self.robot_joint_state = JointState()
        self.robot_joint_angles = JointAngles()
        self.robot_joint_command = JointAngles()
        self.imu_data = Imu()

        # Set data directory (including file prefix)
        self.data_directory = '/home/keno/data/sliding/' + str(rospy.get_time())

        # Previously recorded angles
        self.sliding_data = [[], [], [], []]

        # State descriptors
        self.robot_is_moving = False
        self.robot_has_moved = False

        # Create variables
        self.camera = None
        self.robot = None
        self.group = None

        # Statistics about the plans
        self.plan_stats = []

        rospy.loginfo("Robot Init Successful")

        pass

    def spin(self):
        rospy.spin()

    def receive_pose_data(self, robot_pose):
        self.robot_pose = robot_pose

    def receive_joint_state(self, robot_joint_state):
        self.robot_is_moving = sum(np.abs(robot_joint_state.velocity)) > 0.07
        if self.robot_is_moving:
            self.robot_has_moved = True
        self.robot_joint_state = robot_joint_state

    def receive_joint_command(self, robot_joint_angles):
        self.robot_joint_angles = robot_joint_angles

    def receive_joint_angles(self, robot_joint_command):
        self.robot_joint_command = robot_joint_command

    def receive_imu_data(self, imu_data):
        self.imu_data = imu_data

    def set_camera_flag(self, state):
        self.camera.FLAG = state

    def init_moveit(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.group = moveit_commander.MoveGroupCommander("arm")
        self.group.set_goal_orientation_tolerance(0.0001)

    def get_current_euler(self):
        real_q = self.robot_pose.pose.orientation
        real_euler = euler_from_quaternion(np.array([real_q.x, real_q.y, real_q.z, real_q.w]))

        # do some remapping
        # real_euler = np.array([real_euler[1], real_euler[0] * -1, real_euler[2] + 1.57])
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

    def reset_arm(self):

        position = [0, 0.6, 0.5]

        self.set_arm_position(position, (0, 0, 0))

    def set_arm_position(self, position, euler, force_small_motion=False):

        orientation = quaternion_from_euler(*euler)

        pose_target = Pose()

        pose_target.orientation.x = orientation[0]
        pose_target.orientation.y = orientation[1]
        pose_target.orientation.z = orientation[2]
        pose_target.orientation.w = orientation[3]
        pose_target.position.x = position[0]
        pose_target.position.y = position[1]
        pose_target.position.z = position[2]

        self.group.set_pose_target(pose_target)
        plan_dist = 100
        runs = 0
        threshold = 2

        plans = []
        distances = []

        while plan_dist > threshold:
            plan = self.group.plan()
            points = plan.joint_trajectory.points
            positions = [p.positions for p in points]
            dist = [np.linalg.norm(np.array(p) - np.array(positions[i - 1])) for i, p in enumerate(positions)]
            plan_dist = sum(dist[1:])
            runs += 1
            plans.append(plan)
            distances.append(plan_dist)
            if runs > 1:
                print(plan_dist, threshold)

            if not force_small_motion:
                if len(plans) > 49:
                    plan = plans[np.argmin(distances)]
                    plan_dist = np.min(distances)
                    break
            else:
                if len(plans) > 49:
                    plan_dist = np.min(distances)
                    if plan_dist < 5:
                        plan = plans[np.argmin(distances)]
                        break
                    else:
                        raise ArithmeticError('Could not find appropriate planning solution')

        self.group.execute(plan, wait=False)

        # in case we fuck something up, we need a safety net
        start = rospy.get_time()
        threshold = 0.5
        if not self.robot_is_moving and not self.robot_has_moved:
            while not self.robot_is_moving and not rospy.is_shutdown():
                rospy.sleep(0.01)
                if rospy.get_time() - start > threshold:
                    print("break1")
                    break

        while self.robot_is_moving and not rospy.is_shutdown():
            rospy.sleep(0.01)
            if rospy.get_time() - start > threshold + 10:
                print("break2")
                break

        # collect stats
        self.plan_stats.append({
            'plans': plans,
            'distances': distances
        })

        if self.robot_has_moved:
            self.robot_has_moved = False

        self.group.clear_pose_targets()

        if plan_dist > 1 and force_small_motion:
            return 'recalibrate', pose_target
        return 'default', pose_target

    def rerun_calibration(self):
        rd = RobotData('/home/keno/data/cal_10_1')

        dirpath = rd.base_path + '_rerun'
        os.mkdir(dirpath)

        print("FOUND " + str(len(rd.files)))
        for i in range(len(rd.files)):
            all_joint_pos = rd.load_file_index(i)['robot_joint_angles']
            joint_positions = rd.get_average_joint_position(all_joint_pos)
            print(joint_positions)
            self.move_arm_with_joint_control(joint_positions)
            # todo add proper tracking at some point
            self.collect_pose_data(dirpath, i)

    def collect_pose_data(self, dir_path, index):

        a = index
        collected_data = self.wait_for_data(a)

        with open(dir_path + '/' + str(a) + '.pickle', 'wb') as f:
            print("Got " + str(len(collected_data['robot_pose'])))
            # save data
            pickle.dump(collected_data, f)
            print('Finished Data Collection')

    def wait_for_data(self, a, planned_pose=None, amount=50):

        if not self.REAL_ROBOT_CONNECTED:
            raise EnvironmentError('Connect a real robot to gather data.')

        collected_data = {
            'angle': a,
            'robot_pose': [PoseStamped()],
            'robot_joint_state': [],
            'robot_joint_angles': [],
            'robot_joint_command': [],
            'planned_pose': None,
            'imu_data': [],
            'time': [rospy.get_time(), rospy.get_time()]
        }

        print('Collecting Data')
        collected_data['time'][0] = rospy.get_time()
        while not rospy.is_shutdown() and len(collected_data['robot_pose']) < amount + 1:
            rospy.sleep(0.02)
            if collected_data['robot_pose'][-1].header.seq != self.robot_pose.header.seq:
                collected_data['robot_pose'].append(self.robot_pose)
                collected_data['robot_joint_state'].append(self.robot_joint_state)
                collected_data['robot_joint_angles'].append(self.robot_joint_angles)
                collected_data['robot_joint_command'].append(self.robot_joint_command)
                if self.IMU_CONNTECTED:
                    collected_data['imu_data'].append(self.imu_data)

        # remove initial (empty) state
        collected_data['robot_pose'] = collected_data['robot_pose'][1:]

        # get finished time
        collected_data['time'][1] = rospy.get_time()

        return collected_data

    def analyze_collected_data(self, collected_data):
        print('average pose data')
        orientation = []
        position = []
        for p in collected_data['robot_pose']:
            o = p.pose.orientation
            orientation.append([o.x, o.y, o.z, o.w])
            pos = p.pose.position
            position.append([pos.x, pos.y, pos.z])
        euler = euler_from_quaternion(np.average(orientation, axis=0))
        print([e * (180 / np.pi) for e in euler])
        print(np.average(position, axis=0))
        # print("joint state")
        # print(collected_data['robot_joint_state'][0])
        print("joint angles")
        print(collected_data['robot_joint_angles'][0])
        pass

    def run_calibration(self):
        dirpath = '/home/keno/data/' + str(rospy.get_time())
        os.mkdir(dirpath)

        positions = [[0, 0.6, 0.5], [0, 0.6, 0.5], [0, 0.3, 0.7], [0, 0.5, 0.7]]

        all_angles = [
            # [(0, i * np.pi, 0) for i in np.linspace(0, 0.0555555, 11)],
            [(0, i * np.pi, 0) for i in np.linspace(0, 0.25, 90)],
            [(0, - 1 * i * np.pi, 0) for i in np.linspace(0, 0.5, 5)],
            [(i * np.pi, 0, 0) for i in np.linspace(0, 0.375, 4)],
            [(-1 * i * np.pi, 0, 0) for i in np.linspace(0, 0.375, 4)]
        ]

        # FOR IK X = Y and Y = X
        # AFTER SWITCH new Y (old X) is still negative

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
                    'imu_data': [],
                    'planned_pose': None,
                    'time': [rospy.get_time(), rospy.get_time()]
                }

                print("Going to " + str(max(np.abs(a)) * (180 / np.pi)) + " degree")
                _, collected_data['planned_pose'] = self.set_arm_position(position, a)

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
                        collected_data['imu_data'].append(self.imu_data)

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

            for a in reversed(angles[:-1]):
                print("Resetting to " + str(max(np.abs(a)) * (180 / np.pi)) + " degree")
                self.set_arm_position(position, a)

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

    def connect_to_imu(self):
        topics = [name for (name, _) in rospy.get_published_topics()]

        if '/mavros/imu/data' in topics:
            print("============ FOUND real robot")
            print("============ Subscribing to topics")
            rospy.Subscriber("/mavros/imu/data", Imu, self.receive_imu_data)
            self.IMU_CONNTECTED = True
        else:
            print("============ COULD NOT find real robot")

    def run_single_experiment_ik(self, position, angles):

        print("============ Rotating arm")

        for a in angles:

            rospy.sleep(0.1)
            if self.camera is not None and self.camera.has_slid:
                print("SLIDING RECEIVED, STOPPING")
                # self.group.execute(plan1)
                return "SLIDING"

            print("Going to " + str(max(abs(np.array(a))) * (180 / np.pi)) + " degree")

            # TODO REMOVE IF NO GRAVITY OFFSET IS EXPECTED
            self.camera.expected_offset = max(abs(np.array(a))) * (180 / np.pi)

            status, _ = self.set_arm_position(position, a, force_small_motion=True)
            if status == 'recalibrate':
                self.calibrate_camera()
            # allow camera a glimpse
            rospy.sleep(0.1)

            # self.collect_joint_data_and_add_to_library(a)

            if rospy.is_shutdown():
                exit(0)

        return "DONE"

    def run_real_experiment(self):

        # self.camera = []
        if not self.REAL_ROBOT_CONNECTED or self.camera is None:
            raise EnvironmentError('Camera or Robot not connected')
        # self.camera = None

        positions = [[0, 0.35, 0.5], [0, 0.5, 0.4], [0, 0.4, 0.5], [0, 0.5, 0.4]]

        steps = 50  # 300
        sets = 2000

        all_angles = [
            [(- 1 * i * np.pi, 0, 0) for i in np.linspace(0.0, 0.375, steps)],
            [(0, i * np.pi, 0) for i in np.linspace(0.0, 0.5, steps)],
            [(i * np.pi, 0, 0) for i in np.linspace(0.0, 0.375, steps)],
            [(0, -1 * i * np.pi, 0) for i in np.linspace(0.0, 0.5, steps)]
        ]

        d = 0
        self.set_arm_position(positions[d], (0, 0, 0), force_small_motion=False)
        self.calibrate_camera()

        for i in range(sets):
            t = rospy.get_time()
            print("Going for Run ", i)
            _, d = self.camera.get_next_direction()
            # d = 0
            print("Best Direction is ", d)
            self.set_arm_position(positions[d], (0, 0, 0), force_small_motion=False)

            # just to make sure that we are still going for the right direction
            directions, _ = self.camera.get_next_direction()
            if d not in directions:
                continue

            # wait until we are surely in the right position
            self.calibrate_camera()

            if len(self.sliding_data[d]) < 25:
                angles = all_angles[d]
            else:
                angles = self.get_angles_for_direction(d)

            if self.run_single_experiment_ik(positions[d], angles) == 'SLIDING':
                self.collect_sliding_angles(i, d)
                self.camera.has_slid = False
            print("Set took", rospy.get_time() - t)
            # with open(self.data_directory + '_stats_pickle.plans', 'wb') as f:
            #     pickle.dump(self.plan_stats, f)

        self.set_arm_position(positions[d], (0, 0, 0), force_small_motion=False)
        self.calibrate_camera()

        self.shutdown()

    def calibrate_camera(self):
        if self.camera is None:
            raise EnvironmentError('No camera found')

        self.camera.start_calibration()

        while not rospy.is_shutdown() and self.camera.FLAG == 'CALIBRATE':
            rospy.sleep(1)
            pass

    def collect_sliding_angles(self, run, direction):

        results = []
        eulers = []
        images = []

        for i in range(3):
            rospy.sleep(0.1)
            eulers.append(self.get_current_euler())
            if self.camera is not None:
                images.append(self.camera.current_image)

        eulers = np.average(eulers, axis=0)

        results.append({
            'euler': eulers,
            'time': rospy.get_time(),
            'reference_box': self.camera.reference_box,
            'expected_variance': self.camera.expected_variance,
            'current_box': self.camera.current_box,
            'direction': direction
        })

        self.sliding_data[direction].append(eulers[np.argmax(np.abs(eulers[:-1]))])

        print('FOUND EULERS', eulers)

        with open(self.data_directory + '_run_' + str(run) + '.pickle', 'wb') as f:
            pickle.dump(results, f)

        self.camera.has_slid = False
        pass

    def load_previous_sliding_data(self):

        data = []

        for f in glob.glob('/home/keno/data/sliding/*.pickle'):
            with open(f, 'rb') as f:
                d = pickle.load(f)
                data += [a['euler'] for a in d]

        mapping = [-2, -1, +2, 1]

        angles = [[], [], [], []]
        for d in data:
            # no z axis
            eulers = d[:-1]
            d = mapping.index((np.argmax(np.abs(eulers)) + 1) * np.sign(eulers[np.argmax(np.abs(eulers))]))
            a = eulers[np.argmax(np.abs(eulers))]
            if d == 1:
                a = (np.abs(a) + 0.0226892803) * np.sign(a)
            if d == 3:
                a = (np.abs(a) - 0.0226892803) * np.sign(a)
            if d == 2:
                a = (np.abs(a) + 0.00698132) * np.sign(a)
            if d == 0:
                a = (np.abs(a) - 0.00698132) * np.sign(a)
            angles[d].append(a)

        self.sliding_data = angles

    def get_angles_for_direction(self, direction):

        previous_angles = self.sliding_data[direction]

        avg = np.average(previous_angles)
        std = np.std(previous_angles)

        bounds = [0.375 * np.pi, 0.5 * np.pi, 0.375 * np.pi, 0.5 * np.pi]

        min_angle = max(0, (np.abs(avg) - std * 3)) * np.sign(avg)
        max_angle = min(bounds[direction], (np.abs(avg) + std * 3)) * np.sign(avg)

        print("calculated angle range", direction, min_angle * (180 / np.pi), max_angle * (180 / np.pi))

        if direction == 0 or direction == 2:
            # for some reason this direction has to be flipped
            return [(i, 0, 0) for i in np.linspace(min_angle, max_angle, 50)]
        if direction == 1 or direction == 3:
            return [(0, i, 0) for i in np.linspace(min_angle * -1, max_angle * -1, 50)]

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
