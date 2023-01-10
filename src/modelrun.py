#!/usr/bin/env python
from lib.utils import letterbox_for_img
from tqdm import tqdm
from lib.core.postprocess import morphological_process, connect_lane
from lib.core.function import AverageMeter
from lib.utils import plot_one_box, show_seg_result
from lib.core.general import non_max_suppression, scale_coords
from lib.dataset import LoadImages, LoadStreams
from lib.models import get_net
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.config import update_config
from lib.config import cfg
import PIL.Image as image
import torchvision.transforms as transforms
import numpy as np
import scipy.special
from numpy import random
import torch.backends.cudnn as cudnn
import torch
import cv2
import argparse
import os
import sys
import shutil
import time
from pathlib import Path
import imageio

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()

time1 = 0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        normalize,
    ]
)


def binary_img_to_gray(img):
    color_area = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    color_area[img == 1] = [255]
    return color_area


def get_lanes(img_det):
    global time1
    bs = 1
    img_size = opt.img_size
    h0, w0 = img_det.shape[:2]
    img, ratio, pad = letterbox_for_img(img_det, new_shape=img_size, auto=True)
    h, w = img.shape[:2]
    shapes = (h0, w0), ((h / h0, w / w0), pad)

    img = np.ascontiguousarray(img)

    img = transform(img).to(device)

    img = img.half() if half else img.float()  # uint8 to fp16/32
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    # t1 = time_synchronized()
    # time2 = rospy.Time.now().nsecs*10**-9
    # print("Time before putting in model function {}".format(time2-time1))
    det_out, da_seg_out, ll_seg_out = model(img)
    # time3 = rospy.Time.now().nsecs*10**-9
    # print("Time after putting in the model {}".format(time3-time2))
    # t2 = time_synchronized()
    # if i == 0:
    #     print(det_out)
    inf_out, _ = det_out
    # inf_time.update(t2-t1,img.size(0))

    # Apply NMS
    # t3 = time_synchronized()
    """
    det_pred = non_max_suppression(
        inf_out,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        classes=None,
        agnostic=False,
    )
    """
    # t4 = time_synchronized()

    # nms_time.update(t4-t3,img.size(0))
    # det = det_pred[0]

    # save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")

    _, _, height, width = img.shape
    h, w, _ = img_det.shape
    pad_w, pad_h = shapes[1][1]
    pad_w = int(pad_w)
    pad_h = int(pad_h)
    ratio = shapes[1][0][1]
    ll_predict = ll_seg_out[:, :, pad_h : (height - pad_h), pad_w : (width - pad_w)]
    ll_seg_mask = torch.nn.functional.interpolate(
        ll_predict, scale_factor=int(1 / ratio), mode="bilinear"
    )
    _, ll_seg_mask = torch.max(ll_seg_mask, 1)
    ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
    lanes = binary_img_to_gray(ll_seg_mask)
    # time6 = rospy.Time.now().nsecs*10**-9
    # print("duration between postprocessing and binaryimg to mask {}".format(time6-time5))
    # drivable = binary_img_to_gray(da_seg_mask)
    drivable = None
    return lanes, drivable


def image_callback(data):
    global time1
    try:
        image = bridge.imgmsg_to_cv2(data, "bgr8")
        # image = cv2.cvtColor(image,cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except CvBridgeError as e:
        print(e)
    time1 = rospy.Time.now()
    # print("Time receiving image {}".format(time1))
    lanes, drivable = get_lanes(image)
    """print("Time after get lanes {}".format(
        rospy.Time.now()-time1))"""
    print(f"Time since initial pub {(rospy.Time.now() - time1).to_sec()}")

    try:
        output_pub.publish(bridge.cv2_to_imgmsg(lanes, "mono8"))
    except CvBridgeError as e:
        print(e)


def check():
    img = cv2.imread("inference/images/3c0e7240-96e390d2.jpg")
    cv2.imshow("Img", img)
    lan = get_lanes(img)
    cv2.imshow("Lanes", lan)
    cv2.waitKey(5000)


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    time_array = []
    ini = time.time()
    print(ini)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            lanes, drivable = get_lanes(frame)
            cv2.imshow("Frame", frame)
            cv2.imshow("Detected Lanes", lanes)
            cv2.imshow("Drivable Area", drivable)
        else:
            break
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

        cur = time.time()
        time_array.append(cur - ini)
        ini = cur
    tpf = sum(time_array) / len(time_array)
    fps = 1 / tpf
    print(f"Avg TIme per Frames {tpf}")
    print(f"Avg FPS {fps}")


if __name__ == "__main__":
    rospy.init_node("lane_detection", anonymous=True)
    class Args:
        def __init__(
            self,
            source,
            device,
            save_dir="inference/output",
            iou_tres=0.45,
            img_size=640,
            weights="./weights/End-to-end.pth",
            load_video = False
        ):
            self.source = source
            self.device = device
            self.save_dir = save_dir
            self.iou_tres = iou_tres
            self.img_size = img_size
            self.weights = weights
            self.load_video = load_video

    source = rospy.get_param("~source", "/zed2i/zed_node/rgb/image_rect_color")
    device = rospy.get_param("~device", "cpu")
    opt = Args(source, device)
    print(device)

    if not opt.load_video:
        image_node = opt.source
        output_pub = rospy.Publisher("/lanes", Image, queue_size=20)
        image_sub = rospy.Subscriber(image_node, Image, image_callback)
    else:
        print(f"Video source {opt.source}")

    with torch.no_grad():
        logger, _, _ = create_logger(cfg, cfg.LOG_DIR, "demo")

        device = select_device(logger, opt.device)
        if os.path.exists(opt.save_dir):  # output dir
            shutil.rmtree(opt.save_dir)  # delete dir
        os.makedirs(opt.save_dir)  # make new dir
        half = device.type != "cpu"  # half precision only supported on CUDA

        # Load model
        model = get_net(cfg)
        checkpoint = torch.load(opt.weights, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device)

        if half:
            model.half()
        names = model.module.names if hasattr(model, "module") else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
        _ = (
            model(img.half() if half else img) if device.type != "cpu" else None
        )  # run once
        model.eval()
        if not opt.load_video:
            rospy.loginfo("Waiting for image topics...")
            rospy.spin()
        else:
            load_video(video_path=opt.source)
