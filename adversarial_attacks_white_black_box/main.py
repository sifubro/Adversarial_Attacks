import os
import random
import argparse
import numpy as np
from PIL import Image
from typing import Union
import matplotlib.pyplot as plt

import tensorflow as tf

# from .helper_functions
# from adversarial_attacks_white_black_box.helper_functions

from helper_functions import preprocess, get_imagenet_labels, get_model_pred, decode_predictions, postprocess
from adversarial_attacks import TargetedFGSM, FGSMMaskBackground, ZerothOrderOptimization
from adversarial_attack_base import AdversarialAttack


# GLOBAL VARIABLES
SEED = 1234
IMG_SIZE = (224, 224)

# Set seed for reproducability
os.environ["PYTHONHASHSEED"]= str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



def run_attack(args):
    input_img_path = args.input_img_path
    target_index = args.target_class

    print("Downloading MobileNetV2")
    base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SIZE + (3,), 
                                                include_top = True, 
                                                weights = 'imagenet')
    print("Model Summary")
    base_model.summary()

    print("Freezing model weights")
    base_model.trainable = False

    input_pil_image = Image.open(input_img_path)
    input_image = preprocess(input_pil_image)

    image_array = np.array(input_pil_image)[:,:,:3]
    orig_label, orig_prob, orig_preds = get_model_pred(base_model, image_array, preproc=True)
    print(f"\nPrediction of base_model (before perturbation) on the {input_img_path}")
    print(orig_label, orig_prob)


    y_target = tf.one_hot(target_index, orig_preds.shape[-1])
    y_target = tf.reshape(y_target, (1, orig_preds.shape[-1]))
    _ , target_label, _ = get_imagenet_labels(y_target.numpy())
    print(f"\nReceived target index: {target_index}, i.e. the index of {target_label}")
    print(f"\nRunning iterative FGSM to fool the model into misclassifying the input_image as a {target_label}")

    loss_object = tf.keras.losses.CategoricalCrossentropy()

    if args.attack_method == "FGSM_targeted":
        adversarial_attack_instance = TargetedFGSM(name="FGSM_targeted", 
                                        target_index =target_index,   #254
                                        model = base_model,   #MobileNetV2
                                        criterion  = loss_object,  #CCE
                                        learning_rate=args.learning_rate,  # default= 0.01
                                        sign_grad=args.sign_grad,  # default = True
                                        adv_iterations = args.adv_iterations,  # default =30
                                        num_classes = orig_preds.shape[-1]  # 1000
                                        )
        
    elif args.attack_method == "FGSMMaskBackground":


        adversarial_attack_instance = FGSMMaskBackground(name="FGSMMaskBackground", 
                                        mask_background= bool(args.mask_background),
                                        target_index =target_index,   #254
                                        model = base_model,   #MobileNetV2
                                        criterion  = loss_object,  #CCE
                                        learning_rate=args.learning_rate,  # default= 0.05
                                        sign_grad=args.sign_grad,  # default = True
                                        adv_iterations = args.adv_iterations,  # default =30
                                        num_classes = orig_preds.shape[-1]  # 1000
                                        )
        
    elif args.attack_method == "ZerothOrderOptimization":
        adversarial_attack_instance = ZerothOrderOptimization(name="ZerothOrderOptimization", 
                                        add_noise = args.add_noise, #True
                                        noise_max_val = args.noise_max_val,   #0.01
                                        epsilon= args.epsilon,   #0.05
                                        target_index =target_index,   #254
                                        model = base_model,   #MobileNetV2
                                        criterion  = loss_object,  #CCE
                                        learning_rate=args.learning_rate,  # default= 0.05
                                        sign_grad=args.sign_grad,  # default = True
                                        adv_iterations = args.adv_iterations,  # default =30
                                        num_classes = orig_preds.shape[-1]  # 1000
                                        )
    
    adversarial_attack_instance.run(input_image)

    x_adv = adversarial_attack_instance.retrieve_attack()


    # postprocess from [-1,1] --> [0,255]
    x_adv_array = postprocess(x_adv)[0].numpy()

    # save to disk
    print(f"\nSaving images to {args.output_path}")
    original_image_crop = Image.fromarray(postprocess(input_image[0]).numpy().astype('uint8'), 'RGB')
    adversarial_image_crop = Image.fromarray(postprocess(x_adv)[0].numpy().astype('uint8'), 'RGB')
    noise_added = np.abs(np.array(original_image_crop) - np.array(adversarial_image_crop)).astype(np.float32)


    original_image_crop.save(f"{args.output_path}/original_image_crop.png")
    adversarial_image_crop.save(f"{args.output_path}/adversarial_image_crop.png")


    # Create noise plot
    plt.figure(figsize=(6, 6))
    plt.imshow(noise_added)
    plt.title(f"Noise min={round(np.min(noise_added),2)}, max={round(np.max(noise_added),2)}")
    # Save plot to disk
    plt.savefig(f"{args.output_path}/noise_image_crop.png")
    #Image.fromarray(noise_added, 'RGB').save(f"{args.output_path}/noise_image_crop.png")
    


    print("\nFinished!")








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_img_path', '-i', required=True, type=str, default="adversarial_attacks_white_black_box/cat.jpg", help='path of image file to create adverserial attacks from it')
    parser.add_argument('--output_path', '-o', required=True, type=str , default='static/results_fgsm', help='where to save the adversarial attack')
    parser.add_argument('--target_class', '-t', required=True, type=int, default=254, help='index of the target class in the imagenet classification list. The default index 254 corresponds to a "pug" (dog breed)')
    parser.add_argument('--learning_rate', '-lr', required=False, type=float, default=0.01, help='learning rate for Gradient Descent')
    parser.add_argument('--sign_grad', '-sign', required=False, type=bool , default=True, help='True is using the sign of the gradient for optimization')
    parser.add_argument('--adv_iterations', '-iter', required=False, type=int, default=30, help='How many adversarial iterations to perform')
    parser.add_argument('--mask_background', '-m', required=False, type=bool , default=True, help='True for masking the background in FGSM and performing the attack on the foreground object')
    parser.add_argument('--add_noise', '-an', required=False, type=bool , default=True, help='True adding noise to the adversarial attack for ZOO')
    parser.add_argument('--noise_max_val', '-nv', required=False, type=float , default=0.01, help='Max value for uniform noise to the adversarial attack for ZOO')
    parser.add_argument('--epsilon', '-e', required=False, type=float , default=0.05, help='value for adversarial perturbation to estimate gradients')
    parser.add_argument('--attack_method', '-a', required=False, type=str, default="FGSM_targeted", help='Adversarial attack to run. See adversarial_attacks.py for more')

    args = parser.parse_args()

    run_attack(args)







