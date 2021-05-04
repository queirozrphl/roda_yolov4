# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse
import torch
import time
# from sort import *
# from PIL import Image

"""hyper parameters"""
use_cuda = True

def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    if args.torch:
        m.load_state_dict(torch.load(weightfile))
    else:
        m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    while True:
        val = input("\n numero da imagem: ")
        pred_init_time = time.time()
        named_file = "../fotos_geladeira_4/opencv_frame_" + val + ".png"
        print(named_file)
        img = cv2.imread(named_file)
        # img = cv2.imread(imgfile)
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        for i in range(2):
            start = time.time()
            boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
            finish = time.time()
            if i == 1:
                print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))
        
        plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)
        count_total_in_image(boxes[0], class_names)
        print("\n Total inference time {0} seconds".format(time.time() - pred_init_time))

def detect_cv2_camera(cfgfile, weightfile):
    import cv2
    m = Darknet(cfgfile)
    # mot_tracker = Sort()

    m.print_network()
    m.load_weights(weightfile)
    if args.torch:
        m.load_state_dict(torch.load(weightfile))
    else:
        m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('rtsp://192.168.1.75:8554/mjpeg/1')
    # cap = cv2.VideoCapture("./test.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    print("Starting the YOLO loop...")

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    while True:
        ret, img = cap.read()
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        # piling = Image.fromarray(sized)

        start = time.time() 
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        if boxes is not None:
            # tracked_object = mot_tracker.update(tensorQ)
            finish = time.time()
            print('Predicted in %f seconds.' % (finish - start))
            result_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

        cv2.imshow('Yolo demo', result_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)

#python demo.py -weightfile checkpoints/Yolov4_epoch300.pth -imgfile data/roda_pytorch_yolov4/test/opencv_frame_160_png.rf.d2a36eb6fb827c574db82d42c4cbcb9e.jpg
def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/yolov4.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        default=None,
                        help='path of your image file.', dest='imgfile')
    parser.add_argument('-torch', type=bool, default=False,
                        help='use torch weights')
    args = parser.parse_args()
    

    return args


if __name__ == '__main__':
    args = get_args()
    # print("camera reading")
    # detect_cv2_camera(args.cfgfile, args.weightfile)
    if args.imgfile:
        detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_imges(args.cfgfile, args.weightfile)
        # detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_skimage(args.cfgfile, args.weightfile, args.imgfile)
    else:
        print("capturando pela camera")
        detect_cv2_camera(args.cfgfile, args.weightfile)
