#!/usr/bin/env python
from cv_bridge import CvBridge, CvBridgeError
import argparse
from sensor_msgs.msg import Image
import rospy
import cv2
from time import sleep

bridge = CvBridge()
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str)
parser.add_argument('--pub', type=str)
opt = parser.parse_args()
rospy.init_node('test_node', anonymous=True)
video_path = opt.source
pub_node = opt.pub
print(f"Loading video at {video_path}")
cap = cv2.VideoCapture(video_path)
image_pub = rospy.Publisher(pub_node,Image,queue_size=20)
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        image_pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
