# Adversarial_Attacks description
Create imperceptible pertubations on an input image to fool the model (here mobilenetv2 pretrained on imagenet) to misclassify it as another class (for example an image of a "pug" which is a dog breed at index 254).

### Module explanation
- `adversarial_attack_experimentation.ipynb`: This is a **un-cleaned** development notebook containing basic ideas and code. Needs to be refactored. **For clean scripts and command line interface see below!**
- `adversarial_attack_base.py`: Defines Abstract class (TODO: to convert to ABC) so that each attack can inherit from
- `helper_functions.py`: Basic helper functions for preprocessing input, decoding predictions, postprocessing and visualization. 
- `imagenet_class_list.md`: ImageNet classification indicies for each class (254=pug for example)
- `main.py`: Main script for command line usage


### Experimentation
For all attacks and experimentation done please see the notebook `adversarial_attack_experimentation.ipynb`.

### Main script
For a command line script do the following: 

`pip install requirements.txt`

-----------

**a) Targeted FGSM attack**

This will run the simplest form of attack: `Iterative FGSM targeted attack` to  fool the model into classifying an input image (here of a cat.jpg) to that of a pug (dog breed). The index 254 corresponds to the index of a "pug" in the imagenet dataset (see `imagenet_class_list.md`).

`python main.py --input_img_path ./cat.jpg --target_class 254 --learning_rate 0.01 --sign_grad True --adv_iterations 30`

Results will be saved in `./results_fgsm`


-------------

**b) FGSM attack masking background**

This will run FGSM only on the foregound object (main one) while masking the background during optimization

`python main.py --input_img_path ./cat.jpg --attack_method FGSMMaskBackground --target_class 254 --mask_background True --learning_rate 0.05 --sign_grad True --adv_iterations 10`

Results will be saved in `./results_mask_background`

TODO: Do the reverse
------------


**c) Zeroth Order Optimization Strategy**

This will run a Black Box attack without assuming we have access to the gradients of the model. We estimate the zeroth-order gradient by using 2 perturbed samples.

`python main.py --input_img_path ./cat.jpg  --attack_method ZerothOrderOptimization  --target_class 254 --epsilon 0.05 --learning_rate 0.1 --add_noise True --noise_max_val 0.01 --sign_grad True --adv_iterations 30`

Results will be saved in `./results_zoo`

------------


**d) Natural Evolution Strategies**


TODO

This will run a Black Box attack without assuming we have access to the gradients of the model. We estimate the gradient by using a family of perturbations (e.g. Gaussian)

Results will be saved in `./results_nes`

--------------

**e) FGSM on superpixels**

TODO

Results will be saved in `./results_fgsm_superpixel`

--------------

**For experimentation with more attacks see `adversarial_attack_experimentation.ipynb`**

### TODO list:
- Add additional abstract methods (convert AdversarialAttack class to ABC) so that each attack can inherit from
- experiment more with attacks (e.g. NES, modifying foreground/background and superpixels)
- 
- Introduce `logging` module
- Add `typing` for each function
- `Dockerize` implementation
    - create Dockerfile and run inside a container to ensure same packages
    - serve the methods