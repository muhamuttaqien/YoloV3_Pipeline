import os
import glob
import struct
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.image import imread

import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array

# self-defined library (retrain model)
from yolo3.model import preprocess_true_boxes, yolo_body, yolo_loss
from yolo3.utils import get_random_data
from yolo3.convert_weight import make_yolov3_model, WeightReader

# self-defined library (inference)
from yolo3.yolo import YOLO
from inference.BoundingBox import BoundingBox
from inference.BoundingBoxes import BoundingBoxes
from inference.Evaluator import *
from inference.utils import *

############# dataset related #############

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator.'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data_generator_wrapper that wrap the function data_generator.'''
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

############# model related #############

def get_classes(classes_path):
    '''loads the classes.'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file.'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
        weights_path='model_data/yolo_weights.h5'):
    '''create the training model.'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def load_model(model_weights, anchors_path, classes_path, input_shape):
    """load trained model for performing prediction."""
    
    # define YOLO detector
    yolo = YOLO(**{"model_path": model_weights,
                   "anchors_path": anchors_path,
                   "classes_path": classes_path,
                   "model_image_size" : input_shape})
    
    return yolo

############# inference related #############

def save_prediction(output, image_name, classes_path, save_path):
    """save prediction output into txt file."""
    
    ### get information from output
    bnd_boxes = output[0]
    bnd_scores = output[1]
    bnd_classes = output[2]
    
    class_names = get_classes(classes_path)
    
    # save predictions in save_path
    list_file = open(os.path.join(save_path, f'{image_name}.txt'), 'w')

    for index, bndbox in enumerate(bnd_boxes):

        ymin, xmin, ymax, xmax = bndbox[0], bndbox[1], bndbox[2], bndbox[3]
        
        pred_bnd_name = class_names[bnd_classes[index]]
        
        bnd_score = bnd_scores[index].item()
        pred_bnd_score = round(bnd_score,2)
        
        list_file.write(pred_bnd_name+' '+str(pred_bnd_score)+' '+str(xmin)+' '+str(ymin)+' '+str(xmax)+' '+str(ymax))
        list_file.write('\n')
        
    list_file.close()
    
def get_individual_boundingbox(gt_path, pred_path):
    """read txt files containing bounding boxes (ground truth and detections)."""
    
    allBoundingBoxes = BoundingBoxes()

    # Class representing bounding boxes (ground truths and detections)
    allBoundingBoxes = BoundingBoxes()
    
    ### for ground truths
    file_name = gt_path.split('/')[-1]
    nameOfImage = file_name.replace(".txt", "")
    file_gt = open(gt_path, "r")
    
    for line in file_gt:
        line = line.replace("\n", "")
        if line.replace(' ', '') == '':
            continue
        splitLine = line.split(" ")
        idClass = splitLine[0]  # class
        xmin = float(splitLine[1]) 
        ymin = float(splitLine[2])
        xmax = float(splitLine[3])
        ymax = float(splitLine[4])

        width = xmax-xmin
        height = ymax-ymin

        bb = BoundingBox(
            imageName=nameOfImage,
            classId=idClass,
            x=xmin,
            y=ymin,
            w=width,
            h=height,
            typeCoordinates=CoordinatesType.Absolute, #(200, 200),
            bbType=BBType.GroundTruth,
            format=BBFormat.XYWH)
        allBoundingBoxes.addBoundingBox(bb)
    file_gt.close()
        
    ### for predictions
    file_name = pred_path.split('/')[-1]
    nameOfImage = file_name.replace(".txt", "")
    file_pred = open(pred_path, "r")
    
    for line in file_pred:
        line = line.replace("\n", "")
        if line.replace(' ', '') == '':
            continue
        splitLine = line.split(" ")
        idClass = splitLine[0]  # class
        confidence = float(splitLine[1])  # confidence
        xmin = float(splitLine[2])
        ymin = float(splitLine[3])
        xmax = float(splitLine[4])
        ymax = float(splitLine[5])

        width = xmax-xmin
        height = ymax-ymin

        bb = BoundingBox(
            imageName=nameOfImage,
            classId=idClass,
            x=xmin,
            y=ymin,
            w=width,
            h=height,
            typeCoordinates=CoordinatesType.Absolute,
            bbType=BBType.Detected,
            classConfidence=confidence,
            format=BBFormat.XYWH)
        allBoundingBoxes.addBoundingBox(bb)
    file_pred.close()
    
    return allBoundingBoxes

def visualize_output(image_path, gt_result_path, pred_result_path):
    '''visualize the ground truth and prediction bounding boxes.'''
    
    ### load image
    image = load_img(image_path)
    
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.axis('off')
    ax.imshow(image)
    
    ### for visualization of ground truth
    fh_gt = open(gt_result_path, "r")

    for line in fh_gt:
        line = line.replace("\n", "")
        if line.replace(' ', '') == '':
            continue
        splitLine = line.split(" ")
        idClass = splitLine[0]  # class
        xmin = float(splitLine[1])
        ymin = float(splitLine[2])
        xmax = float(splitLine[3])
        ymax = float(splitLine[4])

        bnd_width = xmax - xmin
        bnd_height = ymax - ymin

        rect = patches.Rectangle((xmin, ymin), bnd_width, bnd_height, 
                                 edgecolor='red', facecolor="none")

        ax.add_patch(rect)

        text = f'{idClass}'
        ax.text(xmax, ymin, text, fontsize=12, color='red')
    
    fh_pred = open(pred_result_path, "r")

    for line in fh_pred:
        line = line.replace("\n", "")
        if line.replace(' ', '') == '':
            continue
        splitLine = line.split(" ")
        idClass = splitLine[0]  # class
        confidence = float(splitLine[1])  # confidence
        xmin = float(splitLine[2])
        ymin = float(splitLine[3])
        xmax = float(splitLine[4])
        ymax = float(splitLine[5])

        bnd_width = xmax - xmin
        bnd_height = ymax - ymin

        rect = patches.Rectangle((xmin, ymin), bnd_width, bnd_height, 
                                 edgecolor='blue', facecolor="none")


        ax.add_patch(rect)

        text = f'{idClass}'
        ax.text(xmax, ymin, text, fontsize=12, color='blue')

        text = f'{confidence}'
        ax.text(xmax, ((ymin+ymax)/2)+10, text, fontsize=12, color='blue')
        
    plt.show()

    fh_gt.close()
    fh_pred.close()
    
def get_individual_score(gt_result_path, pred_result_path):
    '''get individual score for each image.'''
    
    # get score for each image
    boundingboxes = get_individual_boundingbox(gt_result_path, pred_result_path)

    evaluator = Evaluator()

    metricsPerClass = evaluator.GetPascalVOCMetrics(boundingboxes, IOUThreshold=0.3)

    # Loop through classes to obtain their metrics
    precision_self_list = []
    recall_self_list = []
    
    for mc in metricsPerClass:
        # Get metric values per each class
        c = mc['class']
        precision = mc['precision']
        recall = mc['recall']
        average_precision = mc['AP']
        ipre = mc['interpolated precision']
        irec = mc['interpolated recall']
        total_gt = mc['total positives']
        tp = mc['total TP']
        fp = mc['total FP']

        # print precision (self)
        precision_self = tp/(tp+fp)
        precision_self_list.append(precision_self)
        
        print(f'Precision of {c} = {round(precision_self,2)}')

        # print recall (self)
        recall_self = tp/total_gt
        recall_self_list.append(recall_self)
        
        print(f'Recall of {c} = {round(recall_self,2)}')
        
    print('\n')
    
    return precision_self_list, recall_self_list
    
def get_all_boundingboxes(gt_path, pred_path):
    """Read ALL txt files containing bounding boxes (ground truth and detections)."""
    allBoundingBoxes = BoundingBoxes()
    
    # Class representing bounding boxes (ground truths and detections)
    allBoundingBoxes = BoundingBoxes()
    
    # Read ground truths - get all txt files from the gt_path directory
    #folderGT = os.path.join(currentPath, gt_path)
    os.chdir(gt_path)
    gt_files = glob.glob('*.txt')
    gt_files.sort()

    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in gt_files:
        nameOfImage = f.replace('.txt', '')
        fh1 = open(f, 'r')
        for line in fh1:
            line = line.replace('\n', '')
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            xmin = float(splitLine[1]) 
            ymin = float(splitLine[2])
            xmax = float(splitLine[3])
            ymax = float(splitLine[4])
            
            width = xmax-xmin
            height = ymax-ymin

            bb = BoundingBox(
                imageName=nameOfImage,
                classId=idClass,
                x=xmin,
                y=ymin,
                w=width,
                h=height,
                typeCoordinates=CoordinatesType.Absolute, #(200, 200),
                bbType=BBType.GroundTruth,
                format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
        
    # Read detections - get all txt files from the pred_path directory
    os.chdir(pred_path)
    pred_files = glob.glob('*.txt')
    pred_files.sort()
    
    # Read detections from txt file
    # Each line of the files in the detections folder represents a detected bounding box.
    # Each value of each line is  "class_id, confidence, x, y, width, height" respectively
    # Class_id represents the class of the detected bounding box
    # Confidence represents confidence (from 0 to 1) that this detection belongs to the class_id.
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in pred_files:
        # nameOfImage = f.replace("_det.txt","")
        nameOfImage = f.replace('.txt', '')
        # Read detections from txt file
        fh1 = open(f, 'r')
        for line in fh1:
            line = line.replace('\n', '')
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            confidence = float(splitLine[1])  # confidence
            xmin = float(splitLine[2])
            ymin = float(splitLine[3])
            xmax = float(splitLine[4])
            ymax = float(splitLine[5])
            
            width = xmax-xmin
            height = ymax-ymin
            
            bb = BoundingBox(
                imageName=nameOfImage,
                classId=idClass,
                x=xmin,
                y=ymin,
                w=width,
                h=height,
                typeCoordinates=CoordinatesType.Absolute,
                bbType=BBType.Detected,
                classConfidence=confidence,
                format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    return allBoundingBoxes

def get_map(gt_dir, pred_dir):
    
    # Read txt files containing bounding boxes (ground truth and detections)
    boundingboxes = get_all_boundingboxes(gt_dir, pred_dir)
    
    # Create an evaluator object in order to obtain the metrics
    evaluator = Evaluator()

    # Plot Precision x Recall curve
    evaluator.PlotPrecisionRecallCurve(boundingboxes, # Object containing all bounding boxes (ground truths and detections)
                                       IOUThreshold=0.3, # IOU threshold
                                       showAP=True, # Show Average Precision in the title of the plot
                                       showInterpolatedPrecision=False) # Don't plot the interpolated precision curve

    metricsPerClass = evaluator.GetPascalVOCMetrics(boundingboxes, IOUThreshold=0.3)

    mAPs = []

    print(f'Average precision values per class of data:\n')
    # Loop through classes to obtain their metrics
    for mc in metricsPerClass:
        # Get metric values per each class
        c = mc['class']
        precision = mc['precision']
        recall = mc['recall']
        average_precision = mc['AP']
        ipre = mc['interpolated precision']
        irec = mc['interpolated recall']
        total_gt = mc['total positives']
        tp = mc['total TP']
        fp = mc['total FP']

        mAPs.append(average_precision)

        # Print AP per class
        print(f'{c}: {round(average_precision,2)}')

        # print precision (self)
        precision_self = tp/(tp+fp)
        print(f'Precision of {c} = {round(precision_self,2)}')

        # print recall (self)
        recall_self = tp/total_gt
        print(f'Recall of {c} = {round(recall_self,2)}')

    print('Mean Average Precision (mAPs):', round(sum(mAPs)/len(mAPs),2),'\n')
 