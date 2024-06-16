import numpy as np
from PIL import Image
from typing import Union

import tensorflow as tf

def preprocess(image: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    # form batch dimension
    image = tf.expand_dims(image,0) 
    return image

decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

def get_imagenet_labels(probs, topk=1):
    return decode_predictions(probs, top=topk)[0][0]


def get_model_pred(base_model, image, preproc=True):
    if preproc:
        image = preprocess(image)
    preds = base_model.predict(image)
    img_id, label, prob= get_imagenet_labels(preds)
    return label, prob, preds


def postprocess(image: Union[tf.Tensor, np.ndarray]) -> Union[tf.Tensor, np.ndarray]:
    return image*127.5 + 127.5

def display_image(image):
    if type(image) == Image.Image:
        return image
    elif type(image) != np.ndarray:
        image = image.numpy()
    
    if type(image) == np.ndarray:
        return Image.fromarray(image.astype('uint8'), 'RGB')
    else:
        raise TypeError(f"Image is not of type ['PIL.Image', 'np.ndarray', 'tf.Tensor']. Received {image}")