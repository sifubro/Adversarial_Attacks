# Adversarial_Attacks description
Create imperceptible pertubations on an input image to fool the model (here mobilenetv2 pretrained on imagenet) to misclassify it as another class (for example an image of a "pug" which is a dog breed at index 254).

**Best results obtained using the simple Targeted FGSM attack. results in ./adversarial-attacks-white-black-box/results_fgsm**

pypi repo: https://pypi.org/project/adversarial-attacks-white-black-box/0.1.7/

### Usage

-----------

`conda create --name virtenvname python=3.9`

`conda activate virtenvname`

`pip install adversarial-attacks-white-black-box==0.1.7`

-----------

### Running the Docker Container and serve the results in Flask app


1) Build the Docker image:

`sh`

```
docker build -t adversarial_attacks_app .
```

2) Run the Docker container:

`Windows`

```
docker run --name adversarial_attacks_app_container -p 5000:5000 -v %cd%:/app adversarial_attacks_app
```

`Linux`

```
docker run --name adversarial_attacks_app_container -p 5000:5000 -v $(pwd):/app adversarial_attacks_app
```

(In detail: `docker run --name  <container_name> -p  <port_on_host>:<port_on_container>   <name_of_docker_image>`)


3) Open your browser and navigate to `http://localhost:5000` to access the web interface.

------------

### Main script
For a command line script do the following: 


**a) Targeted FGSM attack**

This will run the simplest form of attack: `Iterative FGSM targeted attack` to  fool the model into classifying an input image (here of a cat.jpg) to that of a pug (dog breed). The index 254 corresponds to the index of a "pug" in the imagenet dataset (see `imagenet_class_list.md`).

```
adversarial-attacks --input_img_path ./cat.jpg --output_path ./results_fgsm --target_class 254 --learning_rate 0.01 --sign_grad True --adv_iterations 30
```

Results will be saved in `./results_fgsm`

**Remark** If you are having trouble just go in the subdirectory `adversarial_attacks_white_black_box` and run:

`python main.py --input_img_path ./cat.jpg --target_class 254 --learning_rate 0.01 --sign_grad True --adv_iterations 30`

(remember first to pip install the requirements.txt)

-------------

**b) FGSM attack masking background**

This will run FGSM only on the foregound object (main one) while masking the background during optimization

```
adversarial-attacks --input_img_path ./cat.jpg --output_path ./results_mask_background --attack_method FGSMMaskBackground --target_class 254 --mask_background True --learning_rate 0.05 --sign_grad True --adv_iterations 10
```

Results will be saved in `./results_mask_background`

**TODO:** Do the reverse, i.e. mask the foreground and optimize te background

------------


**c) Zeroth Order Optimization Strategy**

This will run a Black Box attack without assuming we have access to the gradients of the model. We estimate the zeroth-order gradient by using 2 perturbed samples.

```
adversarial-attacks --input_img_path ./cat.jpg --output_path ./results_zoo --attack_method ZerothOrderOptimization  --target_class 254 --epsilon 0.05 --learning_rate 0.1 --add_noise True --noise_max_val 0.01 --sign_grad True --adv_iterations 30
```

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

### Experimentation
For all attacks and experimentation done please see the notebook `adversarial_attack_experimentation.ipynb`. This is a **un-cleaned** development notebook containing basic ideas and code. Needs to be refactored. **For clean scripts and command line interface see below!**

---------------


### Module explanation
- `adversarial_attack_experimentation.ipynb`: This is a **un-cleaned** development notebook containing basic ideas and code. Needs to be refactored. **For clean scripts and command line interface see below!**
- `adversarial_attack_base.py`: Defines Abstract class (TODO: to convert to ABC) so that each attack can inherit from
- `helper_functions.py`: Basic helper functions for preprocessing input, decoding predictions, postprocessing and visualization. 
- `imagenet_class_list.md`: ImageNet classification indicies for each class (254=pug for example)
- `main.py`: Main script for command line usage


-----------

### Nest Steps:
- Add an L1/L2 loss between the original image and perturbation to ensure adversary looks clost to original
- Transfer black box attack. Try to fool another model e.g. ResNet50 and see if the attacks transfer to MobileNetv2
- Experiment more with attacks (e.g. NES, modifying foreground/background and superpixels)

-----------

### TODO list:
- Add additional abstract methods (convert AdversarialAttack class to ABC) so that each attack can inherit from
- Add `unit testing`
- Introduce `logging` module
- Add `typing` for each function
- `Dockerize` implementation
    - create Dockerfile and run inside a container to ensure same packages
    - serve the methods

-----------

### Instructions on how to create the package

`python setup.py sdist bdist_wheel`

`pip install twine`

`twine upload dist/*`

if failed login to PyPI account https://pypi.org/account/login/

and the setting "Add API token" -> copy and configue twine as follows

```
[pypi]

username = your_username

password = API-token
```

(for windows create the following file if it doesn't exist `C:\Users\Username\.pypirc` and add the previous 3 lines)
