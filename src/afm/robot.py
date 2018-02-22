import sys
import rospy
import numpy as np
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from afm.camera import CameraThread
import moveit_commander
import pickle


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
        self.real_pose = PoseStamped().pose

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

    def receive_pose_data(self, robot_data):
        self.real_pose = robot_data.pose

    def set_camera_flag(self, state):
        self.camera.FLAG = state

    def get_current_euler(self):
        real_q = self.real_pose.orientation
        real_euler = euler_from_quaternion(np.array([real_q.x, real_q.y, real_q.z, real_q.w]))

        # do some remapping
        real_euler = np.array([real_euler[1], real_euler[0] * -1, real_euler[2] + 1.57])
        return real_euler

    def get_difference(self, planned_q, planned_coord):
        # temp
        # print(planned_coord, planned_q)

        real_euler = self.get_current_euler()
        real_position = self.real_pose.position

        difference_position = np.array([real_position.x, real_position.y, real_position.z]) - np.array(planned_coord)

        print(difference_position)

        # EULER
        planned_euler = np.array(euler_from_quaternion(planned_q))
        difference_euler = np.array(real_euler - planned_euler)
        print(difference_euler)

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

        return True

    def run_calibration(self):

        positions = [[0, 0.6, 0.5], [0, 0.6, 0.5], [0, 0.4, 0.7], [0, 0.4, 0.7]]

        all_angles = [
            [(0, i * np.pi, 0) for i in np.linspace(0, 0.5, 5)],
            [(0, - 1 * i * np.pi, 0) for i in np.linspace(0, 0.5, 5)],
            [(i * np.pi, 0, 0) for i in np.linspace(0, 0.375, 4)],
            [(-1 * i * np.pi, 0, 0) for i in np.linspace(0, 0.375, 4)]
        ]
        for i in range(4):

            position = positions[i]
            angles = all_angles[i]

            for a in angles:

                q = quaternion_from_euler(*a)

                print("Going to " + str(max(np.abs(a)) * (180 / np.pi)) + " degree")
                self.set_arm_position(position, q)

                rospy.sleep(1)

                if self.REAL_ROBOT_CONNECTED:
                    self.get_difference(q, position)

                # RESET
                self.group.clear_pose_targets()

                if rospy.is_shutdown():
                    exit(0)

            for a in reversed(angles):
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
            print("============ Subscribing to /j2n6s300_driver/out/tool_pose")
            rospy.Subscriber("/j2n6s300_driver/out/tool_pose", PoseStamped, self.receive_pose_data)
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
