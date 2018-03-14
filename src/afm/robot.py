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

    def save_pose_data(self, dir_path, a, collected_data):
        # type: (str, object, dict) -> None
        """
        Dump collected data into a pickle file
        :param collected_data: Data dict that will be saved
        :param dir_path: the data directory
        :param a: index of the current run
        """
        # dump it into a pickle file
        with open(dir_path + '/' + str(a) + '.pickle', 'wb') as f:
            rospy.loginfo("Got " + str(len(collected_data['robot_pose'])))
            # save data
            pickle.dump(collected_data, f)
            rospy.loginfo('Finished Data Collection')
        pass

    def wait_for_data(self, a, planned_pose=None, amount=50):
        # type: (object, Pose, int) -> dict
        """
        Collect data from a variety of sensors including IMU and joint states
        :param planned_pose: The OG pose
        :param a: the angle, only used for identification
        :param amount: how many data points to average over
        """
        # there is no way to collect data if there is no robot connected
        if not self.REAL_ROBOT_CONNECTED:
            raise EnvironmentError('Connect a real robot to gather data.')

        # create a holding dictionary
        collected_data = {
            'angle': a,
            'robot_pose': [PoseStamped()],
            'robot_joint_state': [],
            'robot_joint_angles': [],
            'robot_joint_command': [],
            'planned_pose': planned_pose,
            'imu_data': [],
            'time': [rospy.get_time(), rospy.get_time()]
        }

        rospy.loginfo('Collecting Data')

        # save starting time of data collection
        collected_data['time'][0] = rospy.get_time()

        # capture as many different poses as specified in the parameters
        # collect amount + 1 since there is one pose already in the holding dict
        while not rospy.is_shutdown() and len(collected_data['robot_pose']) < amount + 1:

            # if the last pose is not the same as this one collect this one
            if collected_data['robot_pose'][-1].header.seq != self.robot_pose.header.seq:
                collected_data['robot_pose'].append(self.robot_pose)
                collected_data['robot_joint_state'].append(self.robot_joint_state)
                collected_data['robot_joint_angles'].append(self.robot_joint_angles)
                collected_data['robot_joint_command'].append(self.robot_joint_command)
                if self.IMU_CONNTECTED:
                    collected_data['imu_data'].append(self.imu_data)

            # sleep for a short time since the next pose is probably not imminent
            rospy.sleep(0.02)

        # remove initial (empty) state
        collected_data['robot_pose'] = collected_data['robot_pose'][1:]

        # get finished time
        collected_data['time'][1] = rospy.get_time()

        return collected_data

    @staticmethod
    def analyze_collected_data(collected_data):
        # type: (dict) -> None
        """
        Prints some debug statistics about the collected data
        :param collected_data: data collected via self.wait_for_data
        """
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
        # type: () -> None
        """
        Runs a calibration.
        This is configurable but in general it runs a specific set of angles and collects data while it does so.
        It can be used to compare different angular sensors.
        """

        # Path can be set as required
        dirpath = '/home/keno/data/' + str(rospy.get_time())
        os.mkdir(dirpath)

        # these positions should be adjusted based on the angles the robot will run
        positions = [[0, 0.6, 0.5], [0, 0.6, 0.5], [0, 0.3, 0.7], [0, 0.5, 0.7]]

        all_angles = [
            [(0, i * np.pi, 0) for i in np.linspace(0, 0.25, 90)],
            [(0, - 1 * i * np.pi, 0) for i in np.linspace(0, 0.5, 5)],
            [(i * np.pi, 0, 0) for i in np.linspace(0, 0.375, 4)],
            [(-1 * i * np.pi, 0, 0) for i in np.linspace(0, 0.375, 4)]
        ]

        # for positions
        for i in range(4):

            position = positions[i]
            angles = all_angles[i]

            # run all angles
            for a in angles:

                rospy.loginfo("Going to " + str(max(np.abs(a)) * (180 / np.pi)) + " degree")
                _, planned_pose = self.set_arm_position(position, a)

                # wait for systems to catch up
                rospy.sleep(1)

                # collect data
                collected_data = self.wait_for_data(a, planned_pose)
                self.save_pose_data(dirpath, a, collected_data)

                if rospy.is_shutdown():
                    exit(0)

            # gradually reverse back to the robots original position
            for a in reversed(angles[:-1]):
                rospy.loginfo("Resetting to " + str(max(np.abs(a)) * (180 / np.pi)) + " degree")
                self.set_arm_position(position, a)
                rospy.sleep(1)

        pass

    def connect_to_camera(self):
        # type: () -> None
        """
        Starts the camera thread if the camera is available
        """
        topics = [name for (name, _) in rospy.get_published_topics()]

        if '/raspicam_node/image/compressed' in topics:
            rospy.loginfo("FOUND camera, starting thread")
            self.camera = CameraThread()
            self.camera.start()
            self.set_camera_flag('IGNORE')
        else:
            rospy.logwarn("COULD NOT find camera, running blind")

        pass

    def connect_to_real_robot(self):
        # type: () -> None
        """
        Subscribes to a bunch of topics from the (real) robot
        """
        topics = [name for (name, _) in rospy.get_published_topics()]

        if '/j2n6s300_driver/out/tool_pose' in topics:
            rospy.loginfo("FOUND real robot")
            rospy.Subscriber("/j2n6s300_driver/out/tool_pose", PoseStamped, self.receive_pose_data)
            rospy.Subscriber("/j2n6s300_driver/out/joint_state", JointState, self.receive_joint_state)
            rospy.Subscriber("/j2n6s300_driver/out/joint_angles", JointAngles, self.receive_joint_angles)
            rospy.Subscriber("/j2n6s300_driver/out/joint_command", JointAngles, self.receive_joint_command)
            self.REAL_ROBOT_CONNECTED = True
        else:
            rospy.logwarn("COULD NOT find real robot")

        pass

    def connect_to_imu(self):
        # type: () -> None
        """
        Subscribes to IMU data feed. Useful for debugging angular movements
        """
        topics = [name for (name, _) in rospy.get_published_topics()]

        if '/mavros/imu/data' in topics:
            rospy.loginfo("FOUND IMU")
            rospy.Subscriber("/mavros/imu/data", Imu, self.receive_imu_data)
            self.IMU_CONNTECTED = True
        else:
            rospy.logwarn("COULD NOT IMU")

        pass

    def run_single_experiment_ik(self, position, angles):
        # type: (list or np.array, list or np.array) -> str
        """
        Commands the robot to go to a set of angles using inverse kinematics
        :param position: the position at which these angles should be visited
        :param angles: an list of lists of euler angles in radian
        """

        rospy.loginfo("Starting experiment using inverse kinematics")

        # for all angles
        for a in angles:

            # stop in case of ctrl-c or another error
            if rospy.is_shutdown():
                self.shutdown()

            # let the camera catch up
            rospy.sleep(0.1)

            # if we have a sliding detected, go and stop the experiment
            if self.camera is not None and self.camera.has_slid:
                rospy.loginfo("Sliding received. Stopping")
                # self.group.execute(plan1)
                return "SLIDING"

            rospy.loginfo("Going to " + str(max(abs(np.array(a))) * (180 / np.pi)) + " degree(s)")

            # set the expected gravity offset equal to the angle
            # TODO REMOVE IF NO GRAVITY OFFSET IS EXPECTED
            self.camera.expected_offset = max(abs(np.array(a))) * (180 / np.pi)

            # set the arm position according to the angle
            status, _ = self.set_arm_position(position, a, force_small_motion=True)

            # run recalibration when the planner thinks its nessesary
            if status == 'recalibrate':
                self.calibrate_camera()

            # allow camera a glimpse
            rospy.sleep(0.1)

        return "DONE"

    def run_real_experiment(self):
        # type: () -> None
        """
        Runs an experiment in its length and collects the angle data
        """

        # throw an exception if the camera or the robot is not connected
        if not self.REAL_ROBOT_CONNECTED or self.camera is None:
            raise EnvironmentError('Camera or Robot not connected')

        # these positions are in a specific order that correspond to the angles
        # its also important that the order matches the directions that come from the camera
        positions = [[0, 0.35, 0.5], [0, 0.5, 0.4], [0, 0.4, 0.5], [0, 0.5, 0.4]]

        # how many steps to split the angle range in
        steps = 50
        # how many repetitions to perform
        sets = 2000

        # again all angles in radian
        all_angles = [
            [(- 1 * i * np.pi, 0, 0) for i in np.linspace(0.0, 0.375, steps)],
            [(0, i * np.pi, 0) for i in np.linspace(0.0, 0.5, steps)],
            [(i * np.pi, 0, 0) for i in np.linspace(0.0, 0.375, steps)],
            [(0, -1 * i * np.pi, 0) for i in np.linspace(0.0, 0.5, steps)]
        ]

        # setting the inital direction
        d = 0

        # set the arm position to 0
        self.set_arm_position(positions[d], (0, 0, 0), force_small_motion=False)
        self.calibrate_camera()

        # for each set one repetition
        for i in range(sets):
            # get start time
            t = rospy.get_time()

            rospy.loginfo("Going for Run " + str(i))

            # get the best next direction
            _, d = self.camera.get_next_direction()

            # set arm position to 0 for that direction
            rospy.loginfo("Best Direction is ", d)
            self.set_arm_position(positions[d], (0, 0, 0), force_small_motion=False)

            # just to make sure that the robot is still going for the right direction
            directions, _ = self.camera.get_next_direction()
            if d not in directions:
                continue

            # recalibrate
            self.calibrate_camera()

            # if there is enough data start using the standard deviation for the angle range
            # instead of the angles supplied above
            if len(self.sliding_data[d]) < 25:
                angles = all_angles[d]
            else:
                angles = self.get_angles_for_direction(d)

            # run the experiment with the angles
            if self.run_single_experiment_ik(positions[d], angles) == 'SLIDING':
                # if some sliding happens, collect the angles and reset the camera
                self.collect_sliding_angles(i, d)
                self.camera.has_slid = False
            #
            rospy.loginfo("Set took " + str(rospy.get_time() - t) + ' seconds')

        # reset the arm one more time
        self.set_arm_position(positions[d], (0, 0, 0), force_small_motion=False)

        # shutdown
        self.shutdown()

        pass

    def calibrate_camera(self):
        # type: () -> None
        """
        Sets the camera to calibration mode
        """
        # throw an exception if no camera is detected
        if self.camera is None:
            raise EnvironmentError('No camera found')

        # start the calibration in the camera thread
        self.camera.start_calibration()

        # wait until the calibration is finished
        while not rospy.is_shutdown() and self.camera.FLAG == 'CALIBRATE':
            rospy.sleep(0.1)

        pass

    def collect_sliding_angles(self, run, direction):
        # type: (int, int) -> None
        """
        Collects the sliding angles at a given time
        :param run: index of the run
        :param direction: direction the robot turned in
        """

        # define some empty variables
        results = []
        eulers = []
        images = []

        # collect three eulers and images
        for i in range(3):
            rospy.sleep(0.1)
            eulers.append(self.get_current_euler())
            if self.camera is not None:
                images.append(self.camera.current_image)

        # calculate the average angle
        eulers = np.average(eulers, axis=0)

        # create the results dict
        results.append({
            'euler': eulers,
            'time': rospy.get_time(),
            'reference_box': self.camera.reference_box,
            'expected_variance': self.camera.expected_variance,
            'current_box': self.camera.current_box,
            'direction': direction,
            'movement_timing': [self.camera.start_of_movement, rospy.get_time()]
        })

        # append the angle to the internal angle storage
        self.sliding_data[direction].append(eulers[np.argmax(np.abs(eulers[:-1]))])

        # log some debug info
        rospy.loginfo('Found eulers' + str(eulers))
        rospy.loginfo('Collecting the data took' + str(rospy.get_time() - self.camera.start_of_movement))

        # dump the data into a pickle file
        with open(self.data_directory + '_run_' + str(run) + '.pickle', 'wb') as f:
            pickle.dump(results, f)

        # reset the camera
        self.camera.has_slid = False

        pass

    def load_previous_sliding_data(self):
        # type: () -> None
        """
        Load all previous pickle files to rebuild the angle library
        """

        # define some empty variables
        data = []

        # find all files and load them
        for f in glob.glob('/home/keno/data/sliding/*.pickle'):
            with open(f, 'rb') as f:
                d = pickle.load(f)
                data += [a['euler'] for a in d]

        # this mapping basically maps from the collected data to the internal direction
        mapping = [-2, -1, +2, 1]

        # empty angle library
        angles = [[], [], [], []]

        # reformat data
        for d in data:
            # remove z axis
            eulers = d[:-1]

            # find the direction based on the angle
            d = mapping.index((np.argmax(np.abs(eulers)) + 1) * np.sign(eulers[np.argmax(np.abs(eulers))]))

            # get the max value
            a = eulers[np.argmax(np.abs(eulers))]

            # append it to the library
            angles[d].append(a)

        # set angle library equal to the current angles
        self.sliding_data = angles

        pass

    def get_angles_for_direction(self, direction):
        # type: (int) -> list
        """
        Returns a range of angles based on previous angles
        :param direction: 0 - 3 direction
        """

        # get all previous angles
        previous_angles = self.sliding_data[direction]

        # generate some stats
        avg = np.average(previous_angles)
        std = np.std(previous_angles)

        # these are the upper bounds. Again order matters
        bounds = [0.375 * np.pi, 0.5 * np.pi, 0.375 * np.pi, 0.5 * np.pi]

        # get get the max and min angle
        min_angle = max(0, (np.abs(avg) - std * 3)) * np.sign(avg)
        max_angle = min(bounds[direction], (np.abs(avg) + std * 3)) * np.sign(avg)

        rospy.loginfo("Calculated angle range" + str(min_angle * (180 / np.pi)) + ', ' + str(max_angle * (180 / np.pi)))

        # flip the directions. Just do it. This direction thing needs some time investment anyways
        # return angles according to direction
        if direction == 0 or direction == 2:
            return [(i, 0, 0) for i in np.linspace(min_angle, max_angle, 50)]
        if direction == 1 or direction == 3:
            return [(0, i, 0) for i in np.linspace(min_angle * -1, max_angle * -1, 50)]

    def shutdown(self):
        # type: () -> None
        """
        Shuts the controller and camera down
        """
        rospy.loginfo("Shutting down camera")

        if self.camera is not None:
            self.camera.join()
        rospy.loginfo("Camera shutdown completed")

        # it fails because of a know issue
        # https://github.com/ros-planning/moveit/issues/331
        moveit_commander.roscpp_shutdown()
        rospy.signal_shutdown('Done')
        rospy.loginfo("Everything finished. Now Crashing. RIP")
        return exit(0)
