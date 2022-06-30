import model
from loss import YoloLoss
import tensorflow as tf
from dataset import VOCDataGenerator
import utils
import schedule
import os
import shutil

tf.keras.backend.clear_session()

#### Configuration
train_file = "/home/haris123/python code files/Yolo_implementation/VOCtrainval_06-Nov-2007/VOCdevkit/2007_train.txt"
validation_file = "/home/haris123/python code files/Yolo_implementation/VOCtrainval_06-Nov-2007/VOCdevkit/2007_val.txt"
batch_size = 16

# logs dir
logs_dir = './training_logs'
if os.path.isdir(logs_dir):
        shutil.rmtree(logs_dir)
os.mkdir(logs_dir)

model_dir = './saved_models'
if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
os.mkdir(model_dir)


#### Loading data
X_train, Y_train = utils.open_data_files(train_file)
X_val, Y_val = utils.open_data_files(validation_file)

train_set = VOCDataGenerator(X_train, Y_train, batch_size)
validation_set = VOCDataGenerator(X_val, Y_val, batch_size)

#### loading model
yolo_model = model.get_model()

#### training
lr_callbacks = tf.keras.callbacks.LearningRateScheduler(schedule.lr_scheduler)
logs_callbacks = tf.keras.callbacks.TensorBoard(log_dir=logs_dir)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3, momentum=0.9, rho=0.0005)
loss = YoloLoss(batch_size)

yolo_model.compile(loss=loss, optimizer=optimizer)

yolo_model.fit(train_set, batch_size=batch_size, validation_data=validation_set, epochs=135, 
          callbacks=[lr_callbacks, logs_callbacks])

# saving model
yolo_model.save(model_dir)