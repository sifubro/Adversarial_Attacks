import os
import random
import argparse
import numpy as np
from PIL import Image
from typing import Union

import tensorflow as tf

from helper_functions import preprocess, get_imagenet_labels, get_model_pred, decode_predictions, postprocess
from adversarial_attacks import run_fast_gradient_sign_iterative_method


# GLOBAL VARIABLES
SEED = 1234
IMG_SIZE = (224, 224)

# Set seed for reproducability
os.environ["PYTHONHASHSEED"]= str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



def main(args):
    input_img_path = args.input_img_path
    output_path = args.output_path
    target_index = args.target_class

    print("Downloading MobileNetV2")
    base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SIZE + (3,), 
                                                include_top = True, 
                                                weights = 'imagenet')
    print("Model Summary")
    base_model.summary()

    print("/nFreezing model weights")
    base_model.trainable = False

    input_pil_image = Image.open(input_img_path)
    input_image = preprocess(input_pil_image)

    image_array = np.array(input_pil_image)[:,:,:3]
    orig_label, orig_prob, orig_preds = get_model_pred(base_model, image_array, preproc=True)
    print(f"/nPrediction of base_model (before perturbation) on the {input_img_path}")
    print(orig_label, orig_prob)


    y_target = tf.one_hot(target_index, orig_preds.shape[-1])
    y_target = tf.reshape(y_target, (1, orig_preds.shape[-1]))
    _ , target_label, _ = get_imagenet_labels(y_target.numpy())
    print(f"/nReceived target index: {target_index}, i.e. the index of {target_label}")
    print(f"/nRunning iterative FGSM to fool the model into misclassifying the input_image as a {target_label}")

    
    learning_rate = args.learning_rate # default= 0.01 #0.05
    sign_grad = args.sign_grad # default = True
    adv_iterations = args.adv_iterations # default =30

    loss_object = tf.keras.losses.CategoricalCrossentropy()

    x_adv = run_fast_gradient_sign_iterative_method(input_image,
                                    target_index, 
                                    base_model,
                                    loss_object, 
                                    learning_rate=learning_rate,
                                    sign_grad=sign_grad,
                                    adv_iterations = adv_iterations)




    x_adv_array = postprocess(x_adv)[0].numpy()

    # save to disk
    print(f"/nSaving images to {output_path}")
    Image.fromarray(postprocess(input_image[0]).numpy().astype('uint8'), 'RGB').save(f"{output_path}/original_image_crop.png")
    Image.fromarray(postprocess(x_adv)[0].numpy().astype('uint8'), 'RGB').save(f"{output_path}/adversarial_image_crop.png")

    print("Finished!")








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_img_path', '-i', required=True, type=str, default="./cat.jpg", help='path of image file to create adverserial attacks from it')
    parser.add_argument('--output_path', '-o', required=False, type=str , default='./results_fgsm', help='where to save the adversarial attack')
    parser.add_argument('--target_class', '-t', required=True, type=int, default=254, help='index of the target class in the imagenet classification list. The default index 254 corresponds to a "pug" (dog breed)')
    parser.add_argument('--learning_rate', '-lr', required=False, type=float, default=0.01, help='learning rate for Gradient Descent')
    parser.add_argument('--sign_grad', '-sign', required=False, type=bool , default=True, help='True is using the sign of the gradient for optimization')
    parser.add_argument('--adv_iterations', '-iter', required=False, type=int, default=30, help='How many adversarial iterations to perform')



    args = parser.parse_args()

    main(args)






