import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow.keras.backend as K


class YoloOutput(tf.keras.layers.Layer):
  def __init__(self, shape, **kwargs):
    super().__init__(**kwargs)
    self.target_shape = shape
    
  def get_config(self):
    config = super().get_config()
    config.update({
			"target_shape":self.target_shape
		})
    return config
  
  def call(self, inputs):
    shape = (-1, self.target_shape[0], self.target_shape[1], self.target_shape[2])
    inputs = tf.reshape(inputs, shape)
    # class_scores = tf.keras.activations.softmax(inputs[..., :20]) # class scores
    class_scores = inputs[..., :20]
    conf_boxes = tf.keras.activations.sigmoid(inputs[..., 20:]) # confidence and boxes
    output = tf.concat((class_scores, conf_boxes), axis=-1)
    return output


def block(input, filters, strides):
    """Residual block

    Args:
        input (tensor): this is input coming to block
        filters (int): filters for the kernel
        strides (int): stirdes for convolution 
        skip_layer (bool, optional): This will define whether to use
        convolution for skip layer or not. Defaults to False.
    """
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=strides,
                                   padding='SAME')(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3,
                                   padding='SAME')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    if strides > 1:
      input = tf.keras.layers.Conv2D(filters=filters * 4, kernel_size=1,
                                    padding='SAME', strides=strides)(input)

      x = tf.keras.layers.Conv2D(filters=filters * 4, kernel_size=1,
                                    padding='SAME')(x)
      input = tf.keras.layers.BatchNormalization()(input)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Add()([x, input])
      x = tf.keras.layers.Activation('relu')(x)
      return x      
    else:
      x = tf.keras.layers.Conv2D(filters=filters * 4, kernel_size=1,
                                    padding='SAME')(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Add()([x, input])
      x = tf.keras.layers.Activation('relu')(x)
      return x


def build_network():
    ## resnet 50 backend
    x = tf.keras.Input(shape=(448, 448, 3))
    out = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, 
                      padding='SAME')(x)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation('relu')(out)

    out = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(out)

    out = block(out, filters=64, strides=2)
    out = block(out, filters=64, strides=1)
    out = block(out, filters=64, strides=1)

    out = block(out, filters=128, strides=2)
    out = block(out, filters=128, strides=1)
    out = block(out, filters=128, strides=1)
    out = block(out, filters=128, strides=1)

    out = block(out, filters=256, strides=2)
    out = block(out, filters=256, strides=1)
    out = block(out, filters=256, strides=1)
    out = block(out, filters=256, strides=1)
    out = block(out, filters=256, strides=1)
    out = block(out, filters=256, strides=1)
    
    # out = block(out, filters=512, strides=2)
    # out = block(out, filters=512, strides=1)
    # out = block(out, filters=512, strides=1)

    ## yolo layers     
    out = keras.layers.Conv2D(filters=1024, kernel_size=3, 
                                  padding='SAME')(out)
    out = keras.layers.LeakyReLU(alpha=0.1)(out)
    
    out = keras.layers.Conv2D(filters=1024, kernel_size=3, 
                                  strides=(2, 2), padding='SAME')(out)
    out = keras.layers.LeakyReLU(alpha=0.1)(out)
    
    out = keras.layers.Conv2D(filters=1024, kernel_size=3, 
                                  padding='SAME')(out)
    out = keras.layers.LeakyReLU(alpha=0.1)(out)
    
    out = keras.layers.Conv2D(filters=1024, kernel_size=3, 
                                  padding='SAME')(out)
    out = keras.layers.LeakyReLU(alpha=0.1)(out)
    
    out = keras.layers.Conv2D(filters=1024, kernel_size=7, 
                                  padding='SAME')(out)
    out = keras.layers.LeakyReLU(alpha=0.1)(out)
    
    out = keras.layers.Dropout(rate=0.5)(out)
    out = keras.layers.Flatten()(out)
    out = keras.layers.Dense(4096)(out)
    out = keras.layers.LeakyReLU(alpha=0.1)(out)
    out = keras.layers.Dense(1470)(out)
    out = YoloOutput(shape=(7, 7, 30))(out)
    
    model = tf.keras.models.Model(x, out)
    return model



if __name__=='__main__':
    K.clear_session()
    model = build_network()
    # model.summary()
    