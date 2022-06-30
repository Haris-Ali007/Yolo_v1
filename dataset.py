from tensorflow import keras
import numpy as np
from read_data import read

class VOCDataGenerator(keras.utils.Sequence) :
  
  def __init__(self, images, labels, batch_size) :
    self.images = images
    self.labels = labels
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.images[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

    train_image = []
    train_label = []

    for i in range(0, len(batch_x)):
      img_path = batch_x[i]
      label = batch_y[i]
      image, label_matrix = read(img_path, label)
      train_image.append(image)
      train_label.append(label_matrix)
    return np.array(train_image), np.array(train_label)


if __name__=='__main__':
  train_file = "/home/haris123/python code files/Yolo_implementation/VOCtrainval_06-Nov-2007/VOCdevkit/2007_train.txt"
  with open(train_file, 'r') as file:
    train_x_vals = file.readlines()
  
  X_train = []
  Y_train = []
  idx = 0
  for train_val in train_x_vals:  
    train_val = train_val.replace('\n', '')
    train_labels_images = train_val.split(' ')
    x, y = train_labels_images[0], train_labels_images[1:]
    X_train.append(x)
    Y_train.append(y)
    idx += 1
    if idx == 10:
      break
    
  train_set = VOCDataGenerator(X_train, Y_train, batch_size=2)