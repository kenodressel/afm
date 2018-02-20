#! /usr/bin/env python

import rospy
from sensor_msgs.msg import CompressedImage
import numpy as np
import threading
import cv2


class CameraThread(threading.Thread):

    def __init__(self, queue):
        super(CameraThread, self).__init__()
        self.queue = queue
        self.FLAG = 'IGNORE'
        self.stop_request = threading.Event()
        pass

    def run(self):
        # the Queue here is the GLOBAL_QUEUE from the main thread
        rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, self.receive_camera_data,
                         callback_args=self.queue)
        rospy.spin()

    def receive_camera_data(self, camera_data, main_thread_queue):

        print("DATA FROM CAMERA. STATE " + self.FLAG)

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
            cv2.imwrite('test.jpg', image_np)

            main_thread_queue.put('NO_SLIDE')

            if np.random.random() > 0.90:
                print("FAKING SLIDE")
                self.queue.put('FAKE_SLIDE')
            pass

        elif self.FLAG == 'SHUTDOWN' or self.stop_request:
            rospy.signal_shutdown('END IT')
            self.join()

        else:
            print("GOT UNEXPECTED STATUS " + str(self.FLAG))
            rospy.signal_shutdown('END IT')
            self.join()

    def join(self, timeout=None):
        self.stop_request.set()
        rospy.signal_shutdown('END IT')
        super(CameraThread, self).join(timeout)