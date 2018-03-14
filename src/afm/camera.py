#! /usr/bin/env python

import rospy
from sensor_msgs.msg import CompressedImage
import numpy as np
import threading
import cv2
import time
from sensor_msgs.msg import Image
import cv_bridge


class CameraThread(threading.Thread):
    """
    Camera Thread
    This class contains all camera related code. It receives the image from the camera, processes it and detects movements.
    """

    def __init__(self):
        # Required init function for threads
        super(CameraThread, self).__init__()

        # This flag sets the current action, more about this in has_cube_moved()
        self.FLAG = 'IGNORE'

        # This will be set true when a movement is detected
        self.has_slid = False

        # The position of each corner for the currently detected box
        self.current_box = [(0, 0), (0, 0), (0, 0), (0, 0)]

        # The current image, saved for analytical purposes
        self.current_image = np.array([])

        # The size of the incoming camera feed
        self.image_size = (0, 0)

        # A correction for the offset due to gravitational forces
        self.expected_offset = 0

        # Statistics
        # Some general statistics to analyse the performacne of the code.
        # Not really reliable as this only counts frames that are actually processed when the thread is not blocked.
        # So it works for benchmarking the analysis code but not for the camera feed. Here ROS tools like
        # rostopic info /topic_name should be used
        self.frame_count = 0
        self.start_time = 0
        self.last_time = 0
        self.start_of_movement = 0

        # Initializing the publisher for the processed image
        self.bridge = cv_bridge.CvBridge()
        self.pub = rospy.Publisher('/afm/processed_image', Image, queue_size=5)

        # Some edge calibration related variables
        self.last_box_positions = []
        self.calibration_boxes = []
        self.expected_variance = []
        self.reference_box = []
        pass

    def run(self):
        # type: () -> None
        """
        Subscribe to the incoming camera feed. This can be adjusted to any topic and (image) type.

        The method is also immediately executed when a new CameraThread is started.
        The rospy.spin() is most likely not necessary as the main process will take care of reading the data from the roscore
        but I could not bring myself to take it out and it does no harm.
        """
        rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, self.receive_camera_data)
        rospy.spin()
        pass

    def get_next_direction(self):
        # type: () -> tuple[list, int]
        """
        Provides the next movement direction for the robotic arm based on the relative position of the detected cube.
        """
        # relative distance from the bottom and right hand side
        maxima = np.array(self.image_size) - np.max(self.current_box, axis=0)
        # relative distance from the top and left hand side
        minima = np.min(self.current_box, axis=0)

        # the arrangement in this list depends on the mounting orientation of the camera on the robot.
        directions = np.array([maxima[1], minima[0], minima[1], maxima[0]])

        # consider the best two options based on shortest distance from border
        sorted_dir_index = np.argsort(directions)[:-2]

        # weights have an inverse relation to distances (shorter is better)
        weights = 1.0 / np.array(directions)

        # only consider weights that are relevant
        weights = weights[sorted_dir_index]

        # normalize weights
        weights = np.array(weights / sum(weights))

        # Log some debug info
        rospy.loginfo(directions)
        rospy.loginfo(weights)
        rospy.loginfo(sorted_dir_index)

        # return the indexes of the two best options and a random choice with the weights
        return sorted_dir_index, np.random.choice(sorted_dir_index, 1, p=weights)[0]

    def start_calibration(self):
        # type: () -> None
        """
        Starting the calibration after a short delay to make sure the camera has been properly loaded
        and the first images are in.
        """
        self.get_next_direction()
        # let the camera warm up
        rospy.sleep(0.5)
        rospy.loginfo('Starting Camera Calibration')
        # set flag
        self.FLAG = 'CALIBRATE'
        pass

    def get_box_from_image(self, image):
        # type: (np.array) -> np.array
        """
        Takes image and returns most likely box corners
        :param image: numpy array of grayscale pixels
        """
        # applies a simple binary threshold (less than 30 --> white else black)
        ret, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV)

        # use the image to find the EXTERNAL contours of anything in the image
        _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # if there are no contours its okay for a frame or two so just skip
        if len(cnts) == 0:
            return None

        # get the first contour (there is usually only one)
        c = cnts[0]

        # if there are actually more, find the biggest one.
        # TODO pick the one that is actually closest to the last frame
        if len(cnts) > 1:
            # sort by contour area
            sorted_cnt = sorted(cnts, key=cv2.contourArea, reverse=True)
            c = sorted_cnt[0]

        # fit a rectangle around the found contour
        rect = cv2.minAreaRect(c)

        # instead of using the full curve we are only interested in getting the corner points
        box_points = cv2.boxPoints(rect)

        # there might be some out of bound corners, this replaces the values with either 0 or the max value
        # only works on square images
        corner_issues = 0
        for i, p in enumerate(box_points):
            # bigger than max? replace with max image size
            while max(p) > max(self.image_size):
                box_points[i][np.argmax(p)] = max(self.image_size)
                corner_issues += 1
            # smaller than min? replace with 0
            while min(p) < 0:
                box_points[i][np.argmin(p)] = 0
                corner_issues += 1

        # warn user we got some out of bound corners and had to correct hem
        if corner_issues > 1:
            rospy.logwarn('Corners off bounds: ' + str(corner_issues))
            rospy.logwarn('Sending new distances' + str(box_points))

        return np.array(box_points, dtype=np.int)

    @staticmethod
    def get_distances_between_boxes(box1, box2):
        # type: (np.array, np.array) -> np.array
        """
        Simple distance function between two boxes.

        Its a conservative estimate as it looks at the minimum distance between the corner1 of box1 and corners1-4 of box2.
        This is a lower bound to the distance traveled. It helps to avoid false positives.
        :param box1: First box to compare
        :param box2: Second box to compare against
        """
        distances = []
        for p1 in box1:
            distances.append(min([np.linalg.norm(p1 - p2) for p2 in box2]))

        return np.array(distances)

    def calibrate_edges(self, image):
        # type: (np.array) -> bool
        """
        Used to save the current box into self.reference_box
        Checks the variance of the box over a couple frames and if the variance is below a threshold it returns True
        while it is still calibrating returns False
        :param image: a raw np.array of pixles
        """
        # obtain current box
        box = self.get_box_from_image(image)

        # save box (for statistical use)
        self.current_box = box

        # there can be no "sliding" while calibration
        self.has_slid = False

        # only take boxes with 4 corners
        if len(box) == 4:
            # append box to calibration boxes
            self.calibration_boxes.append(np.array(box, dtype=np.uint64))

            # after gathering 5 frames with boxes look into the variance
            if len(self.calibration_boxes) > 5:

                # calculate average box and expected variance
                self.reference_box = np.average(self.calibration_boxes, axis=0)
                self.expected_variance = np.sum(np.std(self.calibration_boxes, axis=0), axis=1)

                # Reset saved boxes
                self.calibration_boxes = []

                # If the variance of the sample
                if sum(self.expected_variance) > 120:
                    # the variance is too high, need lower variance
                    return False

                return True
        else:
            rospy.logwarn("No Cube detected. Please put the cube on the surface.")

        return False

    def has_cube_moved(self, image):
        # type: (np.array) -> bool
        """
        Detects movement of the current box relative to the reference_box
        :param image: np.array of (grayscale) pixels
        """
        # variables
        corners_reporting_movement = 0

        # get current box
        box = self.get_box_from_image(image)

        # save current box and image for analysis purposes
        self.current_box = box
        self.current_image = image

        # obtain distance of current box to reference box
        distances = self.get_distances_between_boxes(self.reference_box, box)

        # remove the expected variance from the difference between the boxes
        normed_distance = np.abs(distances - self.expected_variance)

        # for each corner check if a movement has occured
        for d in normed_distance:
            # the 10 is just a evaluated threshold
            # the expected_offset comes from the robot function
            # it corrects for the gravitationally caused shift in pixels
            # as of this writing it is the angle (eg 0 at 0 deg, 10 at 10deg etc)
            if d > 10 + self.expected_offset:
                corners_reporting_movement += 1

        # for debug reasons draw the contours on the grayscale image
        cv2.drawContours(image, [box], -1, 255, 5)
        cv2.drawContours(image, [np.array(self.reference_box, dtype=np.int)], -1, 127, 5)

        # get the center of the box
        cX, cY = np.int64(np.average(box, axis=0))

        # draw the center of the box on the image
        try:
            cv2.circle(image, (cX, cY), 6, 255, -1)
            cv2.circle(image, (cX, cY), 6, 255, -1)
        except OverflowError:
            # sometimes an overflow error occurs, I think this is fixed but no guarantee that's why its caught
            # TODO check if it is still a thing
            rospy.logwarn('CAMERA: Received bad coordinates for center drawing.')
            pass

        # Publish the debug image
        self.pub.publish(self.bridge.cv2_to_imgmsg(image, encoding='8UC1'))

        # check if more than two corners report movements
        if corners_reporting_movement > 2:
            rospy.loginfo(str(np.array(normed_distance, np.uint)))
            return True

        return False

    def receive_camera_data(self, camera_data):
        # type: (CompressedImage) -> None
        """
        Callback for subscriber. Handles every frame and decides for appropriate actions. Basically the main function.
        :param camera_data: The callback data from the camera
        """

        # save start time from initial frame
        if self.frame_count == 0:
            self.start_time = time.time()

        # save current time and increment framecount every time
        self.last_time = time.time()
        self.frame_count += 1

        # in case of ignoring incoming messages: exit before the image is even parsed
        if self.FLAG == 'IGNORE':
            return

        # convert image from message to opencv
        np_arr = np.fromstring(camera_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        # this is the case after a slide to get the correct box for the statistics
        if self.FLAG == 'BOX_ONLY':
            self.current_box = self.get_box_from_image(image_np)
            return

        # run calibration until the calibration function returns true, then switch to sliding detection
        if self.FLAG == 'CALIBRATE':

            # set initial data for sliding reference
            self.image_size = image_np.shape
            # reset movement timer
            self.start_of_movement = 0
            if self.calibrate_edges(image_np):
                self.FLAG = 'READY'
                rospy.loginfo('Finished Camera Calibration')
            return

        elif self.FLAG == 'READY':
            # check if movement occured
            if self.has_cube_moved(image_np):
                self.start_of_movement = rospy.get_time()
                self.FLAG = 'BOX_ONLY'
                self.has_slid = True
            return

        elif self.FLAG == 'SHUTDOWN':
            self.join()

        else:
            print("GOT UNEXPECTED STATUS " + str(self.FLAG))
            self.join()

    def join(self, timeout=None):
        # type: (int) -> None
        """
        function gets executed at the end of the thread.
        Reports some statistics, ends rospy and the thread
        :param timeout:
        """
        print("Camera Stats")
        print("Average FPS: " + str(self.frame_count / (self.last_time - self.start_time)))
        print("Uptime: " + str((self.last_time - self.start_time)))
        print("Total Frames: " + str(self.frame_count))
        rospy.signal_shutdown('END IT')
        super(CameraThread, self).join(timeout)
        pass
