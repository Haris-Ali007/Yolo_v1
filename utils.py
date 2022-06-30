import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K


def transform_boxes(box_values):
    """Transform cell coordinates 
    to actual box values in image
    
    Args:
    box_values: size (N, 4) where N is batch.
    
    Returns:
    transformed box_values 
    """
    # formula for reverse transformation
    # x = (x + loc_x) / 7
    # x = x * 448
    # same for all
        
    rows = K.arange(0, 7) # creates a row of indices 0 - 7
    cols = K.arange(0, 7) 
    row_idxs = K.tile(rows, [7]) # creates a tile or row values from 0-7, 7 times

    # expand dims first creates a cols array
    # then it repeats it 7 times giving 7,7 output
    col_idxs = K.tile(K.expand_dims(cols, 0), [7, 1]) 

    # this straightens out col_idxs giving us form similar
    # to row indixes but values arranged in required order
    col_idxs = K.flatten(K.transpose(col_idxs)) 

    # this stacks row and cols and transpose converts them 
    # in required order
    conv_idxs = K.transpose(K.stack([row_idxs, col_idxs]))
    conv_idxs = K.reshape(conv_idxs, shape=(1, 7, 7, 2))
    conv_idxs = K.cast(conv_idxs, K.dtype(box_values))
    xy_values = ((box_values[..., 0:2] + conv_idxs) / [[7, 7]]) * 448
    wh_values = box_values[..., 2:4] * 448
    return xy_values, wh_values



def to_min_max_coords(boxes):
    """Converts box coordinates from
    (x, y, w, h) form to (x_min, y_min,
    x_max, y_max)

    Args:
        boxes: box coordinates of all images 
        (N, s, s, 4)
    Return:
        boxes with coordinates form (x_min, y_min
        x_max, y_max)
    """
    x, y, w, h = boxes[..., 0:1], boxes[..., 1:2], boxes[..., 2:3], boxes[..., 3:4]
    x_min, y_min = np.abs(x - w/2), np.abs(y - h/2)
    x_max, y_max = (x + w/2), (y + h/2) 
    out = np.concatenate((x_min, y_min, x_max, y_max), axis=3)
    return out



def iou(y_pred, y_target):
    """Computes iou scores for predicted and target
    boxes.

    Args:
        y_pred: prediction boxes in form of tensor
        of shape (N, s, s, 4)
        y_target: target boxes in form of tensor
        of shape (N, s, s, 4)

    Returns:
        Iou scores for all the boxes (N, s, s, 1)
    """
    x1 = tf.maximum(y_pred[..., 0:1], y_target[..., 0:1])
    y1 = tf.maximum(y_pred[..., 1:2], y_target[..., 1:2])
    x2 = tf.minimum(y_pred[..., 2:3], y_target[..., 2:3])
    y2 = tf.minimum(y_pred[..., 3:4], y_target[..., 3:4])
    
    # N = y_target.shape[0]
    
    # zero = tf.zeros(shape=x1.shape, dtype=tf.float64)
    intersection = K.maximum(0., (x2 - x1)) * K.maximum(0., (y2 - y1))
    box1_area = tf.abs(y_pred[..., 2:3] - y_pred[..., 0:1]) * tf.abs(y_pred[..., 3:4] 
                                                                    - y_pred[..., 1:2])
    box2_area = tf.abs(y_target[..., 2:3] - y_target[..., 0:1]) * tf.abs(y_target[..., 3:4] 
                                                                    - y_target[..., 1:2])
    union = (box1_area + box2_area) - intersection
    iou = intersection / (union + 1e-2)
    return iou



def nms(boxes, prob_thresh, non_max_thresh):
    """Non max suppression

    Args:
        boxes (list): list of all the 
        predicted boxes.
        
        prob_thresh (float): probability threshold 
        to filter boxes on confidence score.
        
        non_max_thresh (float): non max suppression threshold
    
    Returns:
        Non max supressed boxes.
    """
    bboxes = [box for box in boxes if box[1] > prob_thresh]
    
    # sorting boxes
    bboxes = sorted(bboxes, key= lambda x: x[1], reverse=True)

    bboxes_after_nms = []
    
    while bboxes:
        temp = []
        good_box = bboxes.pop(0)
        box1 = tf.convert_to_tensor(good_box[2:], dtype=tf.float64)
        for box in bboxes:
            box2 = tf.convert_to_tensor(box[2:], dtype=tf.float64)
            score = iou(box1, box2)
            if (score.numpy().item() < non_max_thresh) or (good_box[0] != box[0]):
                temp.append(box)
        bboxes_after_nms.append(good_box)
        bboxes = temp
        
    return bboxes_after_nms

    
    
def open_data_files(file_path):
    """Open data files to load images and labels

    Args:
        file_path (string): path to train and test files
        
    Returns:
        Lists of images and labels.
    """
    with open(file_path, 'r') as file:
        x_vals = file.readlines()
  
    X = []
    Y = []
    
    for vals in x_vals:  
        vals = vals.replace('\n', '')
        label_images = vals.split(' ')
        x, y = label_images[0], label_images[1:]
        X.append(x)
        Y.append(y)
        
    return X, Y


if __name__=="__main__":
    y  = tf.random.uniform((10, 7, 7, 4), 0, 1)
    out = transform_boxes(y)    