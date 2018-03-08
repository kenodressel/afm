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
from afm.pose_library import PoseLibrary
from shape_msgs.msg import SolidPrimitive

# import moveit_msgs.msg
# import geometry_msgs.msg

def norm_q(q):
    a, b, c, d = q
    Z = np.sqrt(a ** 2 + b ** 2 + c ** 2 + d ** 2)
    return np.array([a / Z, b / Z, c / Z, d / Z])


class RobotHandler:

    def __init__(self):

        print("============ started robot init")

        # currently required minimum initialization
        rospy.init_node('afm', anonymous=True)

        self.camera = None
        self.REAL_ROBOT_CONNECTED = False
        self.IMU_CONNTECTED = False
        self.robot_pose = PoseStamped()
        self.robot_joint_state = JointState()
        self.robot_joint_angles = JointAngles()
        self.robot_joint_command = JointAngles()
        self.imu_data = Imu()
        self.data_directory = '/home/keno/data/sliding/' + str(rospy.get_time())

        self.robot = None
        self.group = None

        # pose library
        self.pose_library = PoseLibrary('/home/keno/pose_lib.pickle')

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

        self.group.execute(plan, wait=True)
        self.group.clear_pose_targets()

        if plan_dist > 2 and force_small_motion:
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

    def move_arm_with_joint_control(self, angle_set):

        action_address = '/j2n6s300_driver/joints_action/joint_angles'
        client = actionlib.SimpleActionClient(action_address, ArmJointAnglesAction)
        client.wait_for_server()

        goal = ArmJointAnglesGoal()

        goal.angles.joint1 = angle_set.joint1
        goal.angles.joint2 = angle_set.joint2
        goal.angles.joint3 = angle_set.joint3
        goal.angles.joint4 = angle_set.joint4
        goal.angles.joint5 = angle_set.joint5
        goal.angles.joint6 = angle_set.joint6
        goal.angles.joint7 = angle_set.joint7

        client.send_goal(goal)

        if client.wait_for_result(rospy.Duration(20)):
            # rospy.sleep(1.0)
            return client.get_result()
        else:
            print('the joint angle action timed-out')
            client.cancel_all_goals()

        return None

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

    def collect_joint_data_and_add_to_library(self, a, planned_pose=None):

        if not self.REAL_ROBOT_CONNECTED:
            raise AssertionError('TRYING TO COLLECT DATA WITH NO ROBOT')

        collected_data = self.wait_for_data(a, planned_pose=planned_pose, amount=5)
        self.analyze_collected_data(collected_data)

        # get average positions
        orientation = []
        real_positions = []
        for p in collected_data['robot_pose']:
            o = p.pose.orientation
            orientation.append([o.x, o.y, o.z, o.w])
            pos = p.pose.position
            real_positions.append([pos.x, pos.y, pos.z])

        orientation = np.average(orientation, axis=0)
        real_positions = np.average(real_positions, axis=0)

        p = Pose()
        p.position.x = real_positions[0]
        p.position.y = real_positions[1]
        p.position.z = real_positions[2]
        p.orientation.x = orientation[0]
        p.orientation.y = orientation[1]
        p.orientation.z = orientation[2]
        p.orientation.w = orientation[3]

        joint_arr = np.average([
            [
                s.joint1,
                s.joint2,
                s.joint3,
                s.joint4,
                s.joint5,
                s.joint6,
                s.joint7
            ] for s in collected_data['robot_joint_angles']], axis=0)

        j = JointAngles()
        j.joint1 = joint_arr[0]
        j.joint2 = joint_arr[1]
        j.joint3 = joint_arr[2]
        j.joint4 = joint_arr[3]
        j.joint5 = joint_arr[4]
        j.joint6 = joint_arr[5]
        j.joint7 = joint_arr[6]

        self.pose_library.add_pose_to_library(p, j)

    def build_angle_library(self):

        # self.pose_library.print_pose_index(0)
        # return

        # positions = [[0, 0.6, 0.5], [0, 0.6, 0.5], [0, 0.3, 0.6], [0, 0.5, 0.6]]
        # positions = [[0, 0.4, 0.4], [0, 0.4, 0.4], [0, 0.4, 0.4], [0, 0.4, 0.4]]
        # positions = [[0, 0.5, 0.4], [0, 0.5, 0.4], [0, 0.5, 0.4], [0, 0.5, 0.4]]
        # positions = [[0, 0.4, 0.5], [0, 0.35, 0.5]]
        # positions = [[0, 0.35, 0.5]]
        positions = [[0, 0.5, 0.4], [0, 0.5, 0.4], [0, 0.4, 0.5], [0, 0.35, 0.5]]
        # positions = [[0, 0.5, 0.4], [0, 0.5, 0.4], [-0.4, 0, 0.5], [-0.4, 0, 0.5]]

        steps = 100

        # all_angles = [
        #     # [(0, i * np.pi, 0) for i in np.linspace(0, 0.0555555, 11)],
        #     [(0, i * np.pi, 0) for i in np.linspace(0, 0.5, steps)],
        #     [(0, - 1 * i * np.pi, 0) for i in np.linspace(0, 0.5, steps)],
        #     [(i * np.pi, 0, 0) for i in np.linspace(0, 0.5, steps)],
        #     [(-1 * i * np.pi, 0, 0) for i in np.linspace(0, 0.5, steps)]
        # ]

        all_angles = [
            # [(0, i * np.pi, 0) for i in np.linspace(0, 0.0555555, 11)],
            [(0, i * np.pi, 0) for i in np.linspace(0, 0.5, steps)],
            [(0, - 1 * i * np.pi, 0) for i in np.linspace(0, 0.5, steps)],
            [(i * np.pi, 0, 0) for i in np.linspace(0, 0.375, steps)],
            [(-1 * i * np.pi, 0, 0) for i in np.linspace(0, 0.375, steps)]
        ]

        reset_steps = 5

        all_reverse_angles = [
            # [(0, i * np.pi, 0) for i in np.linspace(0, 0.0555555, 11)],
            [(0, i * np.pi, 0) for i in np.linspace(0, 0.5, reset_steps)],
            [(0, - 1 * i * np.pi, 0) for i in np.linspace(0, 0.5, reset_steps)],
            [(i * np.pi, 0, 0) for i in np.linspace(0, 0.375, reset_steps)],  # 0.375
            [(-1 * i * np.pi, 0, 0) for i in np.linspace(0, 0.375, reset_steps)]
        ]

        # all_angles = [
        #     [(0, 0, 0), (0, 0.1 * np.pi, 0)]
        # ]

        for i in range(4):

            position = positions[i]
            angles = all_angles[i]
            rev_angles = all_reverse_angles[i]
            datas = []
            for a in angles:
                print("Going to " + str(max(np.abs(a)) * (180 / np.pi)) + " degree")
                _, planned_pose = self.set_arm_position(position, a)

                # wait for systems to catch up
                rospy.sleep(0.5)

                self.collect_joint_data_and_add_to_library(a, planned_pose)

            for a in reversed(rev_angles[:-1]):
                print("Resetting to " + str(max(np.abs(a)) * (180 / np.pi)) + " degree")
                self.set_arm_position(position, a)

                rospy.sleep(1)

            # for indx, d in enumerate(datas):
            #     print("Going to pos " + str(indx))
            #
            #     joint_arr = [[s.joint1, s.joint2, s.joint3, s.joint4, s.joint5, s.joint6, s.joint7] for s in
            #                  d['robot_joint_angles']]
            #     avg_joints = {"joint" + str(j_pos + 1): j_data for j_pos, j_data in
            #                   enumerate(np.average(joint_arr, axis=0))}
            #
            #     self.joint_angle_client(avg_joints)
            #     collected_data = self.wait_for_data(indx, amount=10)
            #     self.analyze_collected_data(collected_data)
            #
            # break

        pass

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

    def debug_gathered_data(self):
        all_angles = [
            # [(0, i * np.pi, 0) for i in np.linspace(0, 0.0555555, 11)],
            [(i * np.pi, 0, 0) for i in np.linspace(0, 0.5, 180)],
        ]

        for ac in all_angles:
            for a in ac:
                angles = self.pose_library.get_closest_pose_to_euler_angle(*a)
                self.move_arm_with_joint_control(angles)
                self.collect_joint_data_and_add_to_library(a)
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

    def run_single_experiment(self, position, angles):

        print("============ Rotating arm")

        for a in angles:

            angles = self.pose_library.get_closest_pose_to_euler_angle(*a)

            if self.camera is not None and self.camera.has_slid and self.camera.FLAG == 'READY':
                print("SLIDING RECEIVED, STOPPING")
                # self.group.execute(plan1)
                self.collect_sliding_angles()
                self.camera.has_slid = False
                return "SLIDING"

            print("Going to " + str(max(a) * (180 / np.pi)) + " degree")
            self.move_arm_with_joint_control(angles)

            self.collect_joint_data_and_add_to_library(a)

            if rospy.is_shutdown():
                exit(0)

        return "DONE"

    def run_single_experiment_ik(self, position, angles):

        print("============ Rotating arm")

        for a in angles:

            # angles = self.pose_library.get_closest_pose_to_euler_angle(*a)

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
            rospy.sleep(0.15)

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

        steps = 300
        sets = 100

        all_angles = [
            [(- 1 * i * np.pi, 0, 0) for i in np.linspace(0, 0.375, steps)],
            [(0, i * np.pi, 0) for i in np.linspace(0, 0.5, steps)],
            [(i * np.pi, 0, 0) for i in np.linspace(0, 0.375, steps)],
            [(0, -1 *  i * np.pi, 0) for i in np.linspace(0, 0.5, steps)]
        ]
        d = 0
        self.set_arm_position(positions[d], (0, 0, 0), force_small_motion=False)
        self.calibrate_camera()

        for i in range(sets):
            t = rospy.get_time()
            print("Going for Run ", i)
            d = self.camera.get_next_direction()
            # d = 0
            print("Best Direction is ", d)
            self.set_arm_position(positions[d], (0, 0, 0), force_small_motion=False)

            # just to make sure that we are still going for the right direction
            if d != self.camera.get_next_direction():
                continue

            # wait until we are surely in the right position
            self.calibrate_camera()
            if self.run_single_experiment_ik(positions[d], all_angles[d]) == 'SLIDING':
                self.collect_sliding_angles(i)
                self.camera.has_slid = False
            print("Set took", rospy.get_time() - t)
        self.shutdown()

    def calibrate_camera(self):
        if self.camera is None:
            raise EnvironmentError('No camera found')

        self.camera.start_calibration()

        while not rospy.is_shutdown() and self.camera.FLAG == 'CALIBRATE':
            rospy.sleep(1)
            pass

    def collect_sliding_angles(self, run):

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
            'current_box': self.camera.current_box
        })

        print('FOUND EULERS', eulers)

        with open(self.data_directory + '_run_' + str(run) + '.pickle', 'wb') as f:
            pickle.dump(results, f)

        self.camera.has_slid = False
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
