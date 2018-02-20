#! /usr/bin/env python

import rospy
from sensor_msgs.msg import CompressedImage
import numpy as np
import threading
import cv2
import time


class CameraThread(threading.Thread):

    def __init__(self):
        super(CameraThread, self).__init__()
        self.FLAG = 'IGNORE'
        self.has_slid = False
        self.frame_count = 0
        self.start_time = 0
        self.last_time = 0
        pass

    def run(self):
        rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, self.receive_camera_data)
        rospy.spin()

    def receive_camera_data(self, camera_data):
        if self.frame_count == 0:
            self.start_time = time.time()

        self.last_time = time.time()
        self.frame_count += 1

        if self.FLAG == 'IGNORE':
            return

        np_arr = np.fromstring(camera_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)

        if self.FLAG == 'CALIBRATE':
            # set initial data for sliding reference
            pass

        elif self.FLAG == 'READY':
            # check if movement occured and set SLIDING_DETECTED

            # print(camera_data)
            # cv2.imwrite('test.jpg', image_np)

            if np.random.random() > 0.995:
                print("FAKING SLIDE")
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