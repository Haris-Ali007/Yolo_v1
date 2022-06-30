import tensorflow as tf
import tensorflow.keras.backend as K
from utils import iou


class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, batch_size, lambda_coord= 5, lambda_no_obj = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.lambda_coord = lambda_coord
        self.lambda_no_obj = lambda_no_obj
        self.batch_size = batch_size
        
    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            "batch_size": self.batch_size,
            "lambda_coord": self.lambda_coord,
            "lambd_no_obj": self.lambda_no_obj
        })
        return base_config
    
    def call(self, y_true, y_pred):
        N = self.batch_size
            
        iou_score_box1 = iou(y_pred[..., 21:25], y_true[..., 21:25])
        iou_score_box2 = iou(y_pred[..., 26:30], y_true[..., 21:25])
        scores = tf.concat([iou_score_box1, iou_score_box2], axis=-1)
        
        scores_filter = tf.argmax(scores, axis=-1)
        scores_filter = tf.cast(tf.expand_dims(scores_filter, axis=-1), dtype=tf.float32)
        
        # here we filter out cells where boxes exist
        # this happens on confidence score
        obj_boxes_mask = y_true[..., 20:21]
        
        predicted_boxes = obj_boxes_mask * ( (1 - scores_filter) * y_pred[..., 20:25] + (scores_filter * y_pred[..., 25:30]) )
        
        target_boxes = obj_boxes_mask * y_true[..., 20:25]
        
        #######Object loss#########
        # (x - x_hat)**2 + (y - y_hat)**2 
        sqrd_diff1 = (tf.square(predicted_boxes[..., 1:2] - target_boxes[..., 1:2]) + 
                    tf.square(predicted_boxes[..., 2:3] - target_boxes[..., 2:3]))
        # (sqrt(w) - sqrt(w_hat))**2 + (sqrt(h) - sqrt(y_hat))**2
        # here we signum function to restore signs
        predicted_wh_sqrt = (tf.sign(predicted_boxes[..., 3:5]) * tf.sqrt(tf.abs(predicted_boxes[..., 3:5])))

        target_wh_sqrt = (tf.sign(target_boxes[..., 3:5]) * tf.sqrt( tf.abs(target_boxes[..., 3:5])))

        sqrd_diff2 = (tf.square(predicted_wh_sqrt[..., 0:1] - target_wh_sqrt[..., 0:1]) +
                    tf.square(predicted_wh_sqrt[..., 1:2] - target_wh_sqrt[..., 1:2]))
        obj_loss = self.lambda_coord * (sqrd_diff1 + sqrd_diff2)
        obj_loss = tf.reshape(obj_loss, (N, -1, 1))
        obj_loss = tf.reduce_sum(obj_loss, axis=1)
        
        #######Object confidence loss#########
        obj_confidence_loss = tf.square(predicted_boxes[..., 20:21] - target_boxes[..., 20:21])
        obj_confidence_loss = tf.reshape(obj_loss, (N, -1, 1))
        obj_confidence_loss = tf.reduce_sum(obj_confidence_loss, axis=1)
        
        #######No Object confidence loss#########
        no_obj_pred_conf = ((1 - obj_boxes_mask) * ((1 - scores_filter) * y_pred[..., 20:21] 
                                                + (scores_filter * y_pred[..., 25:26])))
        no_obj_target_conf = (1 - obj_boxes_mask) * y_true[..., 20:21]
        no_obj_loss = tf.square(no_obj_pred_conf - no_obj_target_conf)
        no_obj_loss = tf.reshape(no_obj_loss, (N, -1, 1))
        no_obj_loss = self.lambda_no_obj * tf.reduce_sum(no_obj_loss, axis=1)
        
        #######Class score loss##########
        predicted_classes = obj_boxes_mask * (y_pred[..., :20])
        target_classes = obj_boxes_mask * (y_true[..., :20])
        class_scores = tf.reduce_sum(tf.square(predicted_classes - target_classes), axis=-1)
        class_scores = tf.reshape(class_scores, (N, -1, 1))
        class_score_loss = tf.reduce_sum(class_scores, axis=1)

        ########Total loss########
        loss = obj_loss + no_obj_loss + obj_confidence_loss + class_score_loss
        
        return loss


# lambda_coord= 5
# lambda_no_obj = 0.5
# def yolo_loss(y_true, y_pred):
    
#     N = y_true.shape[0]
    
    
#     iou_score_box1 = iou(y_pred[..., 21:25], y_true[..., 21:25])
#     iou_score_box2 = iou(y_pred[..., 26:30], y_true[..., 21:25])
#     scores = tf.concat([iou_score_box1, iou_score_box2], axis=-1)
    
#     scores_filter = tf.argmax(scores, axis=-1)
#     scores_filter = tf.cast(tf.expand_dims(scores_filter, axis=-1), dtype=tf.float64)
    
#     # here we filter out cells where boxes exist
#     # this happens on confidence score
#     obj_boxes_mask = y_true[..., 20:21]
#     predicted_boxes = obj_boxes_mask * ( (1 - scores_filter) * y_pred[..., 20:25] + (scores_filter * y_pred[..., 25:30]) )
    
#     target_boxes = obj_boxes_mask * y_true[..., 20:25]
    
#     #######Object loss#########
#     # (x - x_hat)**2 + (y - y_hat)**2 
#     sqrd_diff1 = (tf.square(predicted_boxes[..., 1:2] - target_boxes[..., 1:2]) + 
#                 tf.square(predicted_boxes[..., 2:3] - target_boxes[..., 2:3]))
#     # (sqrt(w) - sqrt(w_hat))**2 + (sqrt(h) - sqrt(y_hat))**2
#     # here we signum function to restore signs
#     predicted_wh_sqrt = (tf.sign(predicted_boxes[..., 3:5]) * tf.sqrt(tf.abs(predicted_boxes[..., 3:5])))

#     target_wh_sqrt = (tf.sign(target_boxes[..., 3:5]) * tf.sqrt( tf.abs(target_boxes[..., 3:5])))

#     sqrd_diff2 = (tf.square(predicted_wh_sqrt[..., 0:1] - target_wh_sqrt[..., 0:1]) +
#                 tf.square(predicted_wh_sqrt[..., 1:2] - target_wh_sqrt[..., 1:2]))
#     obj_loss = lambda_coord * (sqrd_diff1 + sqrd_diff2)
#     obj_loss = tf.reshape(obj_loss, (N, -1, 1))
#     obj_loss = tf.reduce_sum(obj_loss, axis=1)
    
#     #######Object confidence loss#########
#     obj_confidence_loss = tf.square(predicted_boxes[..., 20:21] - target_boxes[..., 20:21])
#     obj_confidence_loss = tf.reshape(obj_loss, (N, -1, 1))
#     obj_confidence_loss = tf.reduce_sum(obj_confidence_loss, axis=1)
    
#     #######No Object confidence loss#########
#     no_obj_pred_conf = ((1 - obj_boxes_mask) * ((1 - scores_filter) * y_pred[..., 20:21] 
#                                             + (scores_filter * y_pred[..., 25:26])))
#     no_obj_target_conf = (1 - obj_boxes_mask) * y_true[..., 20:21]
#     no_obj_loss = tf.square(no_obj_pred_conf - no_obj_target_conf)
#     no_obj_loss = tf.reshape(no_obj_loss, (N, -1, 1))
#     no_obj_loss = lambda_no_obj * tf.reduce_sum(no_obj_loss, axis=1)
    
#     #######Class score loss##########
#     predicted_classes = obj_boxes_mask * (y_pred[..., :20])
#     target_classes = obj_boxes_mask * (y_true[..., :20])
#     class_scores = tf.reduce_sum(tf.square(predicted_classes - target_classes), axis=-1)
#     class_scores = tf.reshape(class_scores, (N, -1, 1))
#     class_score_loss = tf.reduce_sum(class_scores, axis=1)

#     ########Total loss########
#     loss = obj_loss + no_obj_loss + obj_confidence_loss + class_score_loss
    
#     return loss


# if __name__=="__main__":
#     y_true = tf.random.uniform((10, 7, 7, 30), 0, 1, dtype=tf.float64)
#     y_pred = tf.random.uniform((10, 7, 7, 30), 0, 1, dtype=tf.float64)
#     print(yolo_loss(y_true, y_pred))
    
 