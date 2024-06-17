import argparse
import logging

from .main import *  # Adjust import as needed



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_img_path', '-i', required=True, type=str, default="./cat.jpg", help='path of image file to create adverserial attacks from it')
    parser.add_argument('--output_path', '-o', required=True, type=str , default='./results_fgsm', help='where to save the adversarial attack')
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


    # Call your main logic here, e.g., run_attack(args)
    run_attack(args)

if __name__ == "__main__":
    main()