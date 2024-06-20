import numpy as np
from PIL import Image
import tensorflow as tf

import onnxruntime as ort
from rembg.bg import remove as rem_bg
from rembg.sessions import u2net

# from .helper_functions
from helper_functions import decode_predictions
from adversarial_attack_base import AdversarialAttack

'''
TODO! Create a Base Class `AdversarialAttack` and subclass each method
Maybe add more methods like:
- estimate_gradient()
- upgrade_adversarial_image()
'''

    
class TargetedFGSM(AdversarialAttack):

    def __init__(self, name="FGSM_targeted", 
                 target_index=-1, 
                 model=None, 
                 criterion=None, 
                 learning_rate=0.01, 
                 sign_grad=True, 
                 adv_iterations=30, 
                 num_classes=1000):
        
        '''
        - target_index: Integer in [0,999] representing the y_target (we want to fool the model in predicting y_target)
        - model: pretrained tensorflow model
        - learning_rate: float for Gradient Descent
        - sign_grad: Boolent. 
            + True if we use the sign of the gradient for optimization. 
            + False if we use the gradient itself
        - adv_iterations: Integer. How can interations of FSGM to perform
        '''
        
        print(f"Instantiating TargetedFGSM for target_index={target_index}")
        self.name = name
        self.x_adv = None #this will be the adversarial attack
        self.target_index = target_index
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.sign_grad = sign_grad
        self.adv_iterations = adv_iterations
        self.num_classes = num_classes
        self.output_path = "./results_fgsm"

        y_target = tf.one_hot(target_index, num_classes)
        self.y_target = tf.reshape(y_target, (1, num_classes))


    def estimate_gradient(self, image):
        NotImplementedError()

    def update_adversarial_attack(self, image):
        NotImplementedError()

    def run(self, input_image: tf.Tensor) -> None:
        '''
        This will estimate the gradient and create the adversarial attack

        - input_image: tf.Tensor of shape (batch, height, width, channels)
        '''

        # create a copy of the input
        x_adv = tf.identity(input_image)

        for iteration in range(self.adv_iterations):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                prediction = self.model(x_adv) #(1, 1000)
                loss = self.criterion(self.y_target, prediction)
                
            print(f"Iteration {iteration} - pred  = {decode_predictions(prediction.numpy(), top=1)[0][0]}")
            print(f"Loss for y_target is {loss}")
            print("="*40)
            
        
            # If we managed to fool the model break the loop
            if np.argmax(prediction) == self.target_index:
                self.x_adv =  x_adv
                break

            # Calculate the gradient of the loss w.r.t image
            gradient = tape.gradient(loss, x_adv)

            if self.sign_grad:
                # get the sign of the gradient
                signed_grad = tf.sign(gradient)
                # Create the adversarial image --> move in the direction of gradient sign 
                x_adv = x_adv - self.learning_rate * signed_grad
            else:
                x_adv = x_adv - self.learning_rate * gradient
                
            x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)

            # if we didn't succeed just return the latest adversarial image..
            if iteration == self.adv_iterations-1:
                print(f"Reached last iteration {self.adv_iterations}. Exiting..")
                self.x_adv =  x_adv
                break

    def retrieve_attack(self) -> tf.Tensor:
        return self.x_adv
    



class FGSMMaskBackground(AdversarialAttack):

    def __init__(self, name="FGSMMaskBackground", 
                 target_index=-1, 
                 model=None, 
                 criterion=None, 
                 mask_background= True,
                 learning_rate=0.05, 
                 sign_grad=True, 
                 adv_iterations=15, 
                 num_classes=1000):
        
        '''
        - mask_background: Boolean. True if you want to mask background and perform the attack on the foreground object only
        - target_index: Integer in [0,999] representing the y_target (we want to fool the model in predicting y_target)
        - model: pretrained tensorflow model
        - epsilon: for estimating
        - learning_rate: float for Gradient Descent
        - sign_grad: Boolent. 
            + True if we use the sign of the gradient for optimization. 
            + False if we use the gradient itself
        - adv_iterations: Integer. How can interations of FSGM to perform
        '''
        if not mask_background:
            raise NotImplementedError("Need to debug this implementation. It introduced many black pixels.")
        
        print(f"Instantiating FGSMMaskBackground for target_index={target_index} with masking background set to {mask_background}")
        self.name = name
        self.mask_background = mask_background
        self.x_adv = None #this will be the adversarial attack
        self.target_index = target_index
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.sign_grad = sign_grad
        self.adv_iterations = adv_iterations
        self.num_classes = num_classes
        if self.mask_background:
            self.output_path = "./results_fgsm_mask_background"
        else:
            self.output_path = "./results_fgsm_mask_foreground"

        y_target = tf.one_hot(target_index, num_classes)
        self.y_target = tf.reshape(y_target, (1, num_classes))

        print("Loading u2net")
        # Load the u2net model with onnxruntime
        try:
            self.u2net_session = ort.InferenceSession("./u2net.onnx")
        except Exception as e:
            print(e)
            print("Downloading model..")
            # download the model if it doesn't exist
            import requests

            def download_model(url, dest_path):
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Ensure the request was successful
                
                with open(dest_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                print(f"Model downloaded and saved to {dest_path}")

            url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
            dest_path = "./u2net.onnx"  # Save the file in the current directory

            download_model(url, dest_path)
            self.u2net_session = ort.InferenceSession("./u2net.onnx")

            
        # Create SessionOptions
        self.options = ort.SessionOptions()
        self.options.intra_op_num_threads = 2
        self.options.inter_op_num_threads = 2
        self.options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        self.options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.options.enable_profiling = False
        self.options.enable_mem_pattern = True
        self.options.enable_cpu_mem_arena = True


    def estimate_gradient(self, image):
        NotImplementedError()

    def update_adversarial_attack(self, image):
        NotImplementedError()


    def remove_background_and_get_mask(self, input_image):
        # set background color to black
        bgcolor = (0,0,0,255)
        
        # Create a new session for the u2net model
        session = u2net.U2netSession(model_name="u2net", sess_opts=self.options)
        
        # Perform the background removal
        result_image = rem_bg(input_image, session=session, bgcolor=bgcolor)
        
        # Get the mask
        masks = session.predict(input_image)
        mask = masks[0]
        
        # Convert the mask to a binary mask (0 for object, 1 for background)
        mask_np = np.array(mask)
        
        # mask background (only object will be "visible")
        # return 0 on the background and 1 ob the foreground object
        binary_mask_bg = np.where(mask_np > 5, 1, 0) 
        
        # mask foreground (only object will be "visible")
        # return 0 on the foreground and 1 ob the background object
        binary_mask_fg = np.where(mask_np > 5, 0, 1)
        
        return result_image, Image.fromarray(binary_mask_bg.astype(np.uint8)), Image.fromarray(binary_mask_fg.astype(np.uint8)) 


    def combine_fg_adv_bg_original(self, input_image, x_adv, mask_background):
        def py_func(input_image, x_adv):
            # postprocess
            input_image_post = input_image[0] * 127.5 + 127.5
            x_adv_post = np.array(x_adv[0]) * 127.5 + 127.5

            x_adv_pil = Image.fromarray(x_adv_post.astype(np.uint8))
            x_adv_pil_no_bg, mask_bg, mask_fg = self.remove_background_and_get_mask(x_adv_pil)

            x_adv_post = np.array(x_adv_pil_no_bg)[:,:,:3]

            
            mask_bg_array = np.array(mask_bg)
            mask_bg_array = np.stack([mask_bg_array,mask_bg_array,mask_bg_array], -1)  #(224,224,3)

            mask_fg_array = np.array(mask_fg)
            mask_fg_array = np.stack([mask_fg_array,mask_fg_array,mask_fg_array], -1)  #(224,224,3)

            if mask_background:
                # combine background or the original image with foreground of attack (x_adv)
                x_adv_post_masked_bg = x_adv_post * mask_bg_array
                x_adv_post_masked_fg = input_image_post * mask_fg_array

                x_adv_post_masked = x_adv_post_masked_bg + x_adv_post_masked_fg
            else:
                # combine background or the original image with foreground of attack (x_adv)
                x_adv_post_masked_bg = x_adv_post * mask_fg_array
                x_adv_post_masked_fg = input_image_post * mask_bg_array

                x_adv_post_masked = x_adv_post_masked_bg + x_adv_post_masked_fg

            #batch it
            x_adv = tf.expand_dims(x_adv_post_masked, 0)
            x_adv = tf.clip_by_value(x_adv, 0., 255.)

            # from [0,255] -> to [-1,1] for the model
            x_adv = (x_adv - 127.5) / 127.5

            mask_bg_array = tf.cast(tf.convert_to_tensor(mask_bg_array), tf.float32)
            mask_fg_array = tf.cast(tf.convert_to_tensor(mask_fg_array), tf.float32)

            return x_adv, mask_bg_array, mask_fg_array

        return tf.py_function(py_func, [input_image, x_adv], [tf.float32, tf.float32, tf.float32])



    def run(self, input_image: tf.Tensor) -> None:
        '''
        This will estimate the gradient and create the adversarial attack

        - input_image: tf.Tensor of shape (batch, height, width, channels)
        '''

        # create a copy of the input
        x_adv = tf.identity(input_image)

        for iteration in range(self.adv_iterations):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                #stop gradients
                input_image = tf.stop_gradient(input_image)
                
                x_adv, mask_bg, mask_fg = self.combine_fg_adv_bg_original(input_image, x_adv, self.mask_background)
                
                mask_bg = tf.stop_gradient(mask_bg)
                mask_fg = tf.stop_gradient(mask_fg)
                
                
                prediction = self.model(x_adv) #(1, 1000)
                loss = self.criterion(self.y_target, prediction)
                
            print(f"Iteration {iteration} - pred  = {decode_predictions(prediction.numpy(), top=1)[0][0]}")
            print(f"Loss for y_target is {loss}")
            print("="*40)
            
        
            # If we managed to fool the model break the loop
            if np.argmax(prediction) == self.target_index:
                self.x_adv = x_adv
                break

            # Calculate the gradient of the loss w.r.t image
            gradient = tape.gradient(loss, x_adv)
            
            if self.mask_background:
                # mask gradient for the background
                gradient = gradient * mask_bg
            else:
                # mask gradient for the foreground
                gradient = gradient * mask_fg
            

            if self.sign_grad:
                # get the sign of the gradient
                signed_grad = tf.sign(gradient)
                # Create the adversarial image --> move in the direction of gradient sign 
                x_adv = x_adv - self.learning_rate * signed_grad
            else:
                x_adv = x_adv - self.learning_rate * gradient
                
            x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)

            # if we didn't succeed just return the latest adversarial image..
            if iteration == self.adv_iterations-1:
                self.x_adv = x_adv
                break

    def retrieve_attack(self) -> tf.Tensor:
        return self.x_adv









class ZerothOrderOptimization(AdversarialAttack):

    def __init__(self, name="ZerothOrderOptimization", 
                 target_index=-1, 
                 model=None, 
                 criterion=None, 
                 add_noise = True,
                 noise_max_val = 0.01, 
                 epsilon= 0.01, 
                 learning_rate=0.01, 
                 sign_grad=True, 
                 adv_iterations=30, 
                 num_classes=1000):
        
        '''
        - add_noise: Boolean. True adding noise to the adversarial attack for ZOO
        - noise_max_val: Max value for uniform noise to the adversarial attack for ZOO
        - epsilon: value for adversarial perturbation to estimate gradients
        - target_index: Integer in [0,999] representing the y_target (we want to fool the model in predicting y_target)
        - model: pretrained tensorflow model
        - learning_rate: float for Gradient Descent
        - sign_grad: Boolent. 
            + True if we use the sign of the gradient for optimization. 
            + False if we use the gradient itself
        - adv_iterations: Integer. How can interations of FSGM to perform
        '''
        
        print(f"Instantiating TargetedFGSM for target_index={target_index}")
        self.name = name
        self.x_adv = None #this will be the adversarial attack
        self.target_index = target_index
        self.model = model
        self.criterion = criterion
        self.add_noise = add_noise
        self.noise_max_val =noise_max_val
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.sign_grad = sign_grad
        self.adv_iterations = adv_iterations
        self.num_classes = num_classes
        self.output_path = "./results_zoo"

        y_target = tf.one_hot(target_index, num_classes)
        self.y_target = tf.reshape(y_target, (1, num_classes))


    def estimate_gradient(self, image):
        NotImplementedError()

    def update_adversarial_attack(self, image):
        NotImplementedError()

    def run(self, input_image: tf.Tensor) -> None:
        '''
        This will estimate the gradient and create the adversarial attack

        - input_image: tf.Tensor of shape (batch, height, width, channels)
        '''

        # create a copy of the input
        x_adv = tf.identity(input_image)

        for iteration in range(self.adv_iterations):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                
                if self.add_noise:
                    noise = tf.random.uniform(x_adv.shape, minval = 0, maxval=self.noise_max_val, dtype=tf.float32, seed=1234)
                    x_adv +=  noise
                
                # get the model prediction
                prediction = self.model(x_adv)
            
                x_pos = tf.identity(x_adv)
                x_neg = tf.identity(x_adv)
                
                # use perturbation of x_adv to estimate gradient
                x_pos += self.epsilon
                x_neg -= self.epsilon
                
                if self.add_noise:
                    # the average of uniform distribution is (max - min)/2 
                    x_pos += self.noise_max_val/2.
                    x_neg -= self.noise_max_val/2.
                
                
                # ensure value in correct range
                x_pos = tf.clip_by_value(x_pos, -1.0, 1.0)
                x_neg = tf.clip_by_value(x_neg, -1.0, 1.0)
                
                
                pred_pos = self.model(x_pos)
                pred_neg = self.model(x_neg)
                
                pos_loss = self.criterion(self.y_target, pred_pos)
                neg_loss = self.criterion(self.y_target, pred_neg)
                
                
                
            print(f"Iteration {iteration}")
            print(f"Pos Loss for y_target is {pos_loss}")
            print(f"Neg Loss for y_target is {neg_loss}")
            

            # Estimate the zeroth-gradient using the 2 samples created
            gradient = (pos_loss - neg_loss) / (2*self.epsilon)

            # Specify what gradient we will use
            if self.sign_grad:
                # get the sign of the gradient
                signed_grad = tf.sign(gradient)
                # Create the adversarial image --> move in the direction of gradient sign 
                x_adv = x_adv - self.learning_rate * signed_grad
            else:
                x_adv = x_adv - self.learning_rate * gradient
                
                
            # ensure correct values (as expected by the model)
            x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
            
            
            #adv_label, adv_prob, adv_preds = self.get_model_pred(x_adv, preproc=False)
            #print(f"Iteration {iteration} - pred  = {adv_label}-{adv_prob}")
            print(f"Iteration {iteration} - pred  = {decode_predictions(prediction.numpy(), top=1)[0][0]}")
            print("="*40)
            
            # If we managed to fool the model break the loop
            if np.argmax(prediction) == self.target_index:
                self.x_adv = x_adv
                break 
            
            # if we didn't succeed just return the latest adversarial image..
            if iteration == self.adv_iterations-1:
                self.x_adv = x_adv
                break 
        

    def retrieve_attack(self) -> tf.Tensor:
        return self.x_adv












