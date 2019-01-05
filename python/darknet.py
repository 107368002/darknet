from ctypes import *
import math
import random
import cv2

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

lib = CDLL("/home/czchen/stevenwork/plate_detection_ML_hw3/darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, cv_image_bgr, thresh=.5, hier_thresh=.5, nms=.45):

    net_width, net_height = lib.network_width(net), lib.network_height(net)
    h,w,c = cv_image_bgr.shape

    h_ratio = h / net_height
    w_ratio = w / net_width

    cv_img = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)
    cv_img = cv2.resize(cv_img,(lib.network_width(net), lib.network_height(net)), interpolation = cv2.INTER_LINEAR)
    im,cv_img = array_to_image(cv_img)
    
    #im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);
    
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                b.x = int(b.x * w_ratio)
                b.y = int(b.y * h_ratio)
                b.w = int(b.w * w_ratio)  
                b.h = int(b.h * h_ratio)
                #print(w_ratio, h_ratio)
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    
    #free_image(im)
    free_detections(dets, num)
    
    return res

def draw_bounding_box(img, label, confidence, x, y, w, h):
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    label = str(label, "utf-8")
    COLORS = {'light_blue':[204,249,0],'light_red':[70,16,234]}
    str_confidence = str(round(confidence*100, 1)) + '%'
    label_confid = str(str(label) + ": " + str_confidence)
    color = COLORS['light_red']

    x_min = int(round((2*x-w)/2))
    x_max = int(round((2*x+w)/2))
    y_min = int(round((2*y-h)/2))
    y_max = int(round((2*y+h)/2))
    max_height, max_width = img.shape[0], img.shape[1]
    if y_max > max_height:
        y_max = max_height
    if x_max > max_width:
        x_max = max_width
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0

    cv2.rectangle(img, (x_min,y_min), (x_max,y_max), color, 1)
    label_distance = 4 # the 'label_distance' pixel that the distance between bbox and the label word.
    label_x_pos, label_y_pos = border_check(x_min,x_max,y_min,y_max, label_distance)
    cv2.putText(img, label_confid, (label_x_pos, label_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return img

def border_check(x_min,x_max,y_min,y_max, label_distance):

    if (x_min - label_distance) > 0:
        if (y_min - label_distance) > 0:
            return x_min, (y_min - label_distance)
        elif (y_min - label_distance) <= 0:
            return x_min, (y_max + label_distance+6)

    elif (x_min - label_distance) <= 0:
        if (y_min - label_distance) > 0:
            return (x_max + label_distance+5), (y_min - label_distance)
        elif (y_min - label_distance) <= 0:
            return (x_max + label_distance+5), (y_max + label_distance+6)

net = None
meta = None

def perform_detect(cfg_path, weights_path, meta_path, cv_img, init=False, verbose=False):
    
    cfg_path = bytes(cfg_path, encoding='utf-8')
    weights_path = bytes(weights_path, encoding='utf-8') 
    meta_path =  bytes(meta_path, encoding='utf-8')
    
    global net
    global meta
    
    if init:
        net = load_net(cfg_path, weights_path, 0)
        meta = load_meta(meta_path)
        return True

    res = detect(net, meta, cv_img)
    if verbose:
        print(res)

    if res == []: # detect nothing
        return cv_img, False


    for label, confidence, (x, y, w, h) in res:
        cv_img = draw_bounding_box(cv_img, label, confidence, x,y,w,h)

    return cv_img, True

def perform_crop(cfg_path, weights_path, meta_path, cv_img, thresh=.5, init=False, crop=False, verbose=False):
    
    cfg_path = bytes(cfg_path, encoding='utf-8')
    weights_path = bytes(weights_path, encoding='utf-8') 
    meta_path =  bytes(meta_path, encoding='utf-8')
    
    global net
    global meta
    if init:
        net = load_net(cfg_path, weights_path, 0)
        meta = load_meta(meta_path)
        return True

    res = detect(net, meta, cv_img, thresh=thresh)
    if verbose:
        print(res)

    if res == []: # detect nothing
        return False

    crop_list = []
    for label, confidence, (x, y, w, h) in res:
        if crop: # return only crop img
            cv_img_copy =cv_img.copy()
            
            x_min = int(round((2*x-w)/2))
            x_max = int(round((2*x+w)/2))
            y_min = int(round((2*y-h)/2))
            y_max = int(round((2*y+h)/2))
            
            max_height, max_width = cv_img_copy.shape[0], cv_img_copy.shape[1]
            if y_max > max_height:
                y_max = max_height
            if x_max > max_width:
                x_max = max_width
            if x_min < 0:
                x_min = 0
            if y_min < 0:
                y_min = 0
            crop_img = cv_img_copy[y_min:y_max, x_min:x_max]
            crop_list.append(crop_img)

    return crop_list

if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    net = load_net("cfg/tiny-yolo.cfg", "tiny-yolo.weights", 0)
    meta = load_meta("cfg/coco.data")
    r = detect(net, meta, "data/dog.jpg")
    print (r)
    

