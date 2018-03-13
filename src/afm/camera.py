#! /usr/bin/env python

import rospy
from sensor_msgs.msg import CompressedImage
import numpy as np
import threading
import cv2
import time
from sensor_msgs.msg import Image
import cv_bridge

"""
Camera Thread
This class contains all camera related code. It receives the image from the camera, processes it and detects movements.
"""


class CameraThread(threading.Thread):

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

        # Initializing the publisher for the processed image
        self.bridge = cv_bridge.CvBridge()
        self.pub = rospy.Publisher('/afm/processed_image', Image, queue_size=5)

        # edge calibration
        self.last_box_positions = []
        self.calibration_boxes = []
        self.expected_variance = []
        self.reference_box = []
        pass

    def run(self):
        rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, self.receive_camera_data)
        rospy.spin()

    def start_calibration(self):
        # let the camera warm up
        rospy.sleep(0.5)
        print('Starting Camera Calibration')
        # set flag
        self.FLAG = 'CALIBRATE'

    def get_next_direction(self):

        maxima = np.array(self.image_size) - np.max(self.current_box, axis=0)
        minima = np.min(self.current_box, axis=0)
        directions = np.array([maxima[1], minima[0], minima[1], maxima[0]])
        # consider the best two options and make a random choice
        sorted_dir_index = np.argsort(directions)[:-2]
        weights = 1.0 / np.array(directions)
        weights = weights[sorted_dir_index]
        weights = np.array(weights / sum(weights))
        print(directions)
        print(weights)
        print(sorted_dir_index)
        return sorted_dir_index, np.random.choice(sorted_dir_index, 1, p=weights)[0]

    def get_box_from_image(self, image):

        ret, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV)
        # self.pub.publish(self.bridge.cv2_to_imgmsg(thresh, encoding='8UC1'))

        # edges = cv2.Canny(image, 50, 100)

        # actually useful stuff
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        # dilated = cv2.dilate(edges, kernel, iterations=2)
        # thresh = dilated

        # not used
        # dilated = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, np.ones((3, 3)), iterations=5)
        # thresh = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=8)

        _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blank = np.zeros_like(image)

        c = cnts[0]

        if len(cnts) > 1:
            sorted_cnt = sorted(cnts, key=cv2.contourArea, reverse=True)
            c = sorted_cnt[0]

        rect = cv2.minAreaRect(c)

        # cv2.drawContours(blank, c, -1, 255, 5)
        box_points = cv2.boxPoints(rect)
        corner_issues = 0
        for i, p in enumerate(box_points):
            while max(p) > max(self.image_size):
                # out of bounds, so we need a new plan
                box_points[i][np.argmax(p)] = max(self.image_size)

                corner_issues += 1

            while min(p) < 0:
                # out of bounds, so we need a new plan
                box_points[i][np.argmin(p)] = 0
                corner_issues += 1

        if corner_issues > 1:
            print('Corners off bounds: ' + str(corner_issues))
            print('Sending new distances' + str(box_points))

        return np.array(box_points, dtype=np.int)

    def get_distances_between_boxes(self, box1, box2):

        # not perfect but since we can expect multiple corners to move a significant amount
        # we dont need to reliably match all points. potentially multiple points map to one but
        # then assume 0 for unmapped point. We really want to avoid false positives

        distances = []
        for p1 in box1:
            distances.append(min([np.linalg.norm(p1 - p2) for p2 in box2]))

        return np.array(distances)

    def calibrate_edges(self, image):

        box = self.get_box_from_image(image)
        self.current_box = box

        # No movement during calibration
        self.has_slid = False

        if len(box) == 4:
            if len(self.calibration_boxes) == 0:
                self.calibration_boxes.append(np.array([np.array(c) for c in box]))
            else:
                # sorted_box = [None, None, None, None]
                # for p in box:
                #     closest = np.argmin([np.linalg.norm(p - p_o) for p_o in self.calibration_boxes[0]])
                #     sorted_box[closest] = p
                # self.calibration_boxes.append(np.array(sorted_box, dtype=np.uint64))
                self.calibration_boxes.append(np.array(box, dtype=np.uint64))

            if len(self.calibration_boxes) > 5:
                # min_box = np.min(self.calibration_boxes, axis=0)
                max_box = np.max(self.calibration_boxes, axis=0)

                self.reference_box = np.average(self.calibration_boxes, axis=0)
                self.expected_variance = np.sum(np.std(self.calibration_boxes, axis=0), axis=1)

                print(self.reference_box)
                print(self.expected_variance)

                # reset old variables
                self.calibration_boxes = []

                if sum(self.expected_variance) > 120:
                    # the variance is too high, need lower variance
                    return False

                return True
        else:
            print("No Cube detected. Please put the cube on the surface.")

        return False

    def has_cube_moved(self, image):

        box = self.get_box_from_image(image)
        self.current_box = box
        self.current_image = image

        corners_reporting_movement = 0

        distances = self.get_distances_between_boxes(self.reference_box, box)
        normed_distance = np.abs(distances - self.expected_variance)

        for d in normed_distance:
            if d > 10 + self.expected_offset:
                corners_reporting_movement += 1

        # if sum(normed_distance) > 50 + self.expected_offset * 4:
        #     print("forcing it")
        #     corners_reporting_movement = 4

        self.last_box_positions.append(box)

        # print(np.array(normed_distance, np.int), corners_reporting_movement, 10 + self.expected_offset / 1.5)

        if len(self.last_box_positions) > 5:
            self.last_box_positions.pop(0)

        cv2.drawContours(image, [box], -1, 255, 5)
        cv2.drawContours(image, [np.array(self.reference_box, dtype=np.int)], -1, 127, 5)

        # cX, cY = np.int0(np.average(box, axis=0) - np.average(self.reference_box, axis=0))
        # print(cX, cY)
        cX, cY = np.int0(np.average(box, axis=0))
        # draw the contour and center of the shape on the image

        try:
            cv2.circle(image, (cX, cY), 6, 255, -1)
            cv2.circle(image, (cX, cY), 6, 255, -1)
        except OverflowError:
            pass
            # TODO do something about this
            # print(cX, cY)
            # print(box)

        self.pub.publish(self.bridge.cv2_to_imgmsg(image, encoding='8UC1'))

        if corners_reporting_movement > 2:
            print(np.array(normed_distance, np.uint), corners_reporting_movement, self.expected_offset)
            return True
        else:
            box = np.array(np.average(self.last_box_positions, axis=0), np.uint)

        return False

    def receive_camera_data(self, camera_data):

        if self.frame_count == 0:
            self.start_time = time.time()

        self.last_time = time.time()
        self.frame_count += 1

        if self.FLAG == 'IGNORE':
            return

        np_arr = np.fromstring(camera_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        if self.FLAG == 'BOX_ONLY':
            self.current_box = self.get_box_from_image(image_np)
            return

        if self.FLAG == 'CALIBRATE':
            # set initial data for sliding reference

            self.image_size = image_np.shape
            if self.calibrate_edges(image_np):
                self.FLAG = 'READY'
                print('Finished Camera Calibration')
            return

        elif self.FLAG == 'READY':
            # check if movement occured and set SLIDING_DETECTED

            # print(camera_data)
            # cv2.imwrite('test.jpg', image_np)

            # self.get_box_with_ar(image_np)
            if self.has_cube_moved(image_np):
                # print("DETECTED SLIDE")
                self.FLAG = 'BOX_ONLY'
                self.has_slid = True
            return

        elif self.FLAG == 'SHUTDOWN':
            self.join()

        else:
            print("GOT UNEXPECTED STATUS " + str(self.FLAG))
            self.join()

    def join(self, timeout=None):
        print("Camera Stats")
        print("Average FPS: " + str(self.frame_count / (self.last_time - self.start_time)))
        print("Uptime: " + str((self.last_time - self.start_time)))
        print("Total Frames: " + str(self.frame_count))
        rospy.signal_shutdown('END IT')
        super(CameraThread, self).join(timeout)
