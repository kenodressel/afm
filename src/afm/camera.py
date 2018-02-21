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

    def __init__(self):
        super(CameraThread, self).__init__()
        self.FLAG = 'IGNORE'
        self.has_slid = False
        # stats
        self.frame_count = 0
        self.start_time = 0
        self.last_time = 0
        # debug view
        self.bridge = cv_bridge.CvBridge()
        self.pub = rospy.Publisher('/afm/processed_image', Image, queue_size=5)
        # color calibration
        self.calibration_range = []
        self.cube_colors = (-1, -1)
        self.last_frame = np.array([])
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
        rospy.sleep(1)
        print('Starting Camera Calibration')
        # set flag
        self.FLAG = 'CALIBRATE'

    def use_color_calibration(self, image):

        x, y = image.shape
        s = 30  # size

        calibration_spot = image[x / 2 - s:x / 2 + s, y / 2 - s:y / 2 + s].copy()
        cv2.rectangle(image, (y / 2 + s, x / 2 + s), (y / 2 - s, x / 2 - s), 255, 2)

        calibration_spot_f32 = image[x / 2 - s:x / 2 + s, y / 2 - s:y / 2 + s].astype(np.float32)
        # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
        criteria = (cv2.cv.CV_TERMCRIT_EPS + cv2.cv.CV_TERMCRIT_ITER, 10, 1.0)

        # Set flags (Just to avoid line break in the code)
        flags = cv2.KMEANS_RANDOM_CENTERS

        # Apply KMeans
        compactness, labels, centers = cv2.kmeans(calibration_spot_f32.reshape((-1, 1)), 2, criteria, 1, flags)
        label_img = np.array(labels).reshape((s * 2, s * 2)).astype(np.uint8) * 255
        # image_np[x/2 - s:x/2 + s, y/2 - s:y/2 + s] = label_img

        self.pub.publish(self.bridge.cv2_to_imgmsg(image, encoding='8UC1'))
        print("Ratio: " + str(float(sum(labels)) / len(labels)))
        if float(sum(labels)) / len(labels) > 0.7:
            # 1 is dominant
            points_in_cube = [p for i, p in enumerate(calibration_spot.reshape(-1)) if labels[i] == 1]
            pass
        elif float(sum(labels)) / len(labels) < 0.3:
            points_in_cube = [p for i, p in enumerate(calibration_spot.reshape(-1)) if labels[i] == 0]
            pass

        else:
            print("Could not properly find one dominant color. Please try to stay in the square!")
            return

        counts = {u: points_in_cube.count(u) for u in set(points_in_cube)}

        # TODO work with counts to ignore color range outliers

        print([np.min(points_in_cube), np.max(points_in_cube)])
        self.calibration_range.append([np.min(points_in_cube), np.max(points_in_cube)])

        max_frames = 100
        print("Calibration: " + str(self.frame_count * 100 / max_frames) + "%")

        if len(self.calibration_range) > max_frames:
            # collected enough data

            self.cube_colors = np.average(self.calibration_range, axis=0)
            print(self.cube_colors)
            self.FLAG = 'READY'

    def get_box_from_image(self, image):

        edges = cv2.Canny(image, 50, 100)

        # edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((3, 3)), iterations=1)

        # self.pub.publish(self.bridge.cv2_to_imgmsg(edges, encoding='8UC1'))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # thresh = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=8)

        thresh = dilated

        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_SIMPLE)

        blank = np.zeros_like(image)

        c = cnts[0]

        if len(cnts) > 1:
            sorted_cnt = sorted(cnts, key=cv2.contourArea, reverse=True)
            c = sorted_cnt[0]

        rect = cv2.minAreaRect(c)

        # cv2.drawContours(blank, c, -1, 255, 5)

        return np.array(cv2.cv.BoxPoints(rect), np.uint)

    def get_distances_between_boxes(self, box1, box2):

        # not perfect but since we can expect multiple corners to move a significant amount
        # we dont need to reliably match all points. potentially multiple points map to one but
        # then assume 0 for unmapped point. We really want to avoid false positives

        distances = [0, 0, 0, 0]

        for p2 in box2:
            temp_distances = [np.linalg.norm(p2 - p1) for p1 in box1]
            closest = np.argmin(temp_distances)
            distances[closest] = temp_distances[closest]

        return np.array(distances)

    def calibrate_edges(self, image):

        box = self.get_box_from_image(image)

        if len(box) > 0:
            if len(self.calibration_boxes) == 0:
                self.calibration_boxes.append(box)
            else:
                sorted_box = [None, None, None, None]
                for p in box:
                    closest = np.argmin([np.linalg.norm(p - p_o) for p_o in self.calibration_boxes[0]])
                    sorted_box[closest] = p
                self.calibration_boxes.append(sorted_box)

            if len(self.calibration_boxes) > 20:
                # min_box = np.min(self.calibration_boxes, axis=0)
                max_box = np.max(self.calibration_boxes, axis=0)

                self.reference_box = np.average(self.calibration_boxes, axis=0)
                self.expected_variance = self.get_distances_between_boxes(self.reference_box, max_box)

                print(self.reference_box)
                print(self.expected_variance)

                return True
        else:
            print("No Cube detected. Please put the cube on the surface.")

        return False

    def has_cube_moved(self, image):

        box = self.get_box_from_image(image)

        corners_reporting_movement = 0

        distances = self.get_distances_between_boxes(self.reference_box, box)
        normed_distance = distances - self.expected_variance

        for d in normed_distance:
            if d > 10:
                corners_reporting_movement += 1

        self.last_box_positions.append(box)

        if len(self.last_box_positions) > 5:
            self.last_box_positions.pop(0)

        cv2.drawContours(image, [box], -1, 255, 5)

        cX, cY = np.int0(np.average(box, axis=0))
        # print(cX, cY)

        # draw the contour and center of the shape on the image
        cv2.circle(image, (cX, cY), 6, 255, -1)

        self.pub.publish(self.bridge.cv2_to_imgmsg(image, encoding='8UC1'))

        if corners_reporting_movement > 2:
            print(np.array(normed_distance, np.uint), corners_reporting_movement)
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
        image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_GRAYSCALE)

        if self.FLAG == 'CALIBRATE':
            # set initial data for sliding reference

            # self.use_color_calibration(image_np)

            if self.calibrate_edges(image_np):
                self.FLAG = 'READY'
                print('Finished Camera Calibration')
            pass

        elif self.FLAG == 'READY':
            # check if movement occured and set SLIDING_DETECTED

            # print(camera_data)
            # cv2.imwrite('test.jpg', image_np)

            if self.has_cube_moved(image_np):
                print("DETECTED SLIDE")
                self.has_slid = True
            pass

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
