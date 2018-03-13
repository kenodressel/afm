import glob
import sys

import rospy
import numpy as np
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState, Imu
from kinova_msgs.msg import JointAngles
from afm.camera import CameraThread
import moveit_commander
import pickle
import os


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

    @staticmethod
    def spin():
        # type: () -> None
        """
        For debug only. Runs the spin command to frequently receive updates for all topics
        """
        rospy.spin()
        pass

    def receive_pose_data(self, robot_pose):
        # type: (PoseStamped) -> None
        """
        Callback for /j2n6s300_driver/out/tool_pose
        :param robot_pose: Data from Publisher
        """
        self.robot_pose = robot_pose
        pass

    def receive_joint_state(self, robot_joint_state):
        # type: (JointState) -> None
        """
        Callback for /j2n6s300_driver/out/joint_state. Also determines if the robot is moving
        :param robot_joint_state: Data from Publisher
        """
        # If the sum of all joint velocities is over a threshold the robot state will be determined as moving
        self.robot_is_moving = sum(np.abs(robot_joint_state.velocity)) > 0.07

        # once we detected a movement, set a flag
        if self.robot_is_moving:
            self.robot_has_moved = True
        # Save joints tate to robot
        self.robot_joint_state = robot_joint_state

    def receive_joint_command(self, robot_joint_angles):
        # type: (JointAngles) -> None
        """
        Callback for /j2n6s300_driver/out/joint_angles
        :param robot_joint_angles: Data from Publisher
        """
        self.robot_joint_angles = robot_joint_angles
        pass

    def receive_joint_angles(self, robot_joint_command):
        # type: (JointAngles) -> None
        """
        Callback for /j2n6s300_driver/out/joint_command
        :param robot_joint_command: Data from Publisher
        """
        self.robot_joint_command = robot_joint_command
        pass

    def receive_imu_data(self, imu_data):
        # type: (Imu) -> None
        """
        Callback for /mavros/imu/data
        :param imu_data: Data from Publisher
        """
        self.imu_data = imu_data
        pass

    def set_camera_flag(self, state):
        # type: (str) -> None
        """
        Sets the camera flag.
        :param state: String in CAPS declaring the state
        """
        # TODO check if status is allowed
        self.camera.FLAG = state
        pass

    def init_moveit(self):
        # type: () -> None
        """
        Initializes the inverse kinematics system. Required if inverse kinematics or the simulation is going to be used
        """
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.group = moveit_commander.MoveGroupCommander("arm")

        # this sets the allowed goal error for planning to 1/10 of the original threshold
        self.group.set_goal_orientation_tolerance(0.0001)
        pass

    def get_current_euler(self):
        # type: () -> np.array
        """
        Takes current pose and returns euler angles
        """
        real_q = self.robot_pose.pose.orientation
        real_euler = euler_from_quaternion(np.array([real_q.x, real_q.y, real_q.z, real_q.w]))
        return real_euler

    def get_difference(self, planned_q, planned_coord):
        # type: (np.array, np.array) -> tuple[np.array, np.array]
        """
        Debugging method used to get live pose error
        :param planned_q: Planned quaternion angles
        :param planned_coord: Planned cartesian coordinates
        """

        # gather real data from the current states
        real_euler = self.get_current_euler()
        real_position = self.robot_pose.pose.position

        # get difference in position
        difference_position = np.array([real_position.x, real_position.y, real_position.z]) - np.array(planned_coord)

        # get difference in euler
        planned_euler = np.array(euler_from_quaternion(planned_q))
        difference_euler = np.array(real_euler - planned_euler)

        return difference_position, difference_euler

    def set_arm_position(self, position, euler, force_small_motion=False):
        # type: (list or np.array, list or np.array, bool) -> tuple[str, Pose]
        """
        Takes angles and position and guides robot to correct position using inverse kinematics
        :param position: list of cartesian coordinates (XYZ)
        :param euler: lsit of euler angles (XYZ)
        :param force_small_motion: True to throw an error if no small motion is possible
        """
        # convert euler to quaternion
        orientation = quaternion_from_euler(*euler)

        # create empty target pose
        pose_target = Pose()

        # set pose to desired target
        pose_target.orientation.x = orientation[0]
        pose_target.orientation.y = orientation[1]
        pose_target.orientation.z = orientation[2]
        pose_target.orientation.w = orientation[3]
        pose_target.position.x = position[0]
        pose_target.position.y = position[1]
        pose_target.position.z = position[2]

        # pass target pose to inverse kinematics
        self.group.set_pose_target(pose_target)

        # the planning distances is 100 to make sure we run at least one iteration of the while loop
        plan_dist = 100

        # run counter
        runs = 0

        # threshold (value got from experience)
        threshold = 2

        # empty arrays for stats collection and final choice of plan
        plans = []
        distances = []

        # iterate while distance of calcualted plan is too high
        while plan_dist > threshold:
            # use inverse kinematics to obtain a plan
            plan = self.group.plan()
            # extract all intermediate points from a plan
            points = plan.joint_trajectory.points
            # extract all positions in a list
            positions = [p.positions for p in points]
            # calculate distance between previous and current point
            dist = [np.linalg.norm(np.array(p) - np.array(positions[i - 1])) for i, p in enumerate(positions)]
            # sum to obtain total travel distance
            plan_dist = sum(dist[1:])

            # gather some data
            runs += 1
            plans.append(plan)
            distances.append(plan_dist)

            # if we have enough plans we can assume we got a good one
            if len(plans) > 49:
                # get the best plan
                plan = plans[np.argmin(distances)]
                # including its distance
                plan_dist = np.min(distances)

                # if we still have not found a good plan, raise an exception
                if plan_dist > 5 and force_small_motion:
                    raise ArithmeticError('Could not find appropriate planning solution')

                # otherwise break the loop
                break

        # we execute the best plan we found (in async)
        self.group.execute(plan, wait=False)

        # this is a hack to keep the thread running while the robot moves
        # capture the start time
        start = rospy.get_time()
        # first threshold to wait until the robot starts moving (usually it has already moved by this time)
        threshold = 0.5
        # assuming the robot did not move before and its not moving now, wait for the robot to move or until the threshold
        # is reached
        if not self.robot_is_moving and not self.robot_has_moved:
            while not self.robot_is_moving and not rospy.is_shutdown():
                # this allows the camera to work through frames
                rospy.sleep(0.01)
                if rospy.get_time() - start > threshold:
                    rospy.loginfo("Ran into timeout for waiting until the robot moves. Assuming the motion was missed.")
                    break

        # here it can be assumed that we are moving
        while self.robot_is_moving and not rospy.is_shutdown():
            # this allows the camera to work through frames
            rospy.sleep(0.01)
            # the threshold here is 10 seconds. The robot is expected to never move more than 10 seconds.
            if rospy.get_time() - start > threshold + 10:
                rospy.logwarn("Ran into timeout for waiting while the robot moves. Assuming the motion is stuck.")
                break

        # collect stats
        self.plan_stats.append({
            'plans': plans,
            'distances': distances
        })

        # reset robot state
        if self.robot_has_moved:
            self.robot_has_moved = False

        # clear the ik targets
        self.group.clear_pose_targets()

        # if the robot got a bigger motion than expected
        # it forces the system to recalibrate itself after the motion
        if plan_dist > 1 and force_small_motion:
            return 'recalibrate', pose_target

        # otherwise just return the normal state
        return 'default', pose_target

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
