# Adversarial_Attacks description
Create imperceptible pertubations on an input image to fool the model (here mobilenetv2 pretrained on imagenet) to misclassify it as another class (for example an image of a "pug" which is a dog breed at index 254).

### Experimentation
For all attacks and experimentation done please see the notebook `adversarial_attack_experimentation.ipynb`.

### Main script
For a command line script do the following: 

`pip install requirements.txt`

`python main.py --input_img_path ./cat.jpg --output_path ./results --target_class 254 --learning_rate 0.01 --sign_grad True --adv_iterations 30`

At the moment this will run the simplest form of attack: `Iterative FGSM targeted attack` to  fool the model into classifying an input image (here of a cat.jpg) to that of a pug (dog breed). The index 254 corresponds to the index of a "pug" in the imagenet dataset (see `imagenet_class_list.md`).

**For experimentation with more attacks see `adversarial_attack_experimentation.ipynb`**

### TODO list:
- Create a class (`AdversarialAttack`) and subclass for each method
- experiment more with attacks (e.g. NES, modifying foreground/background and superpixels)
- 
- Introduce `logging` module
- Add `typing` for each function
- `Dockerize` implementation
    - create Dockerfile and run inside a container to ensure same packages
    - serve the methods