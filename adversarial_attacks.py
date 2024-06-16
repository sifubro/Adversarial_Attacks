import numpy as np
from PIL import Image
import tensorflow as tf

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
    




#TODO! Create a subclass of AdversarialAttack
def run_zero_order_optimization_method(input_image,
                                        target_index, 
                                        model,
                                        loss_object, 
                                        add_noise = True, 
                                        noise_max_val = 0.01,
                                        learning_rate=0.1,
                                        epsilon = 0.05,
                                        sign_grad=True,
                                        adv_iterations = 30):
    '''
    - input_image: tf.Tensor of shape (batch, height, width, channels)
    - target_index: Integer in [0,999] representing the y_target (we want to fool the model in predicting target_index)
    - model: pretrained tensorflow model
    - learning_rate: float for Gradient Descent
    - sign_grad: Boolean. 
        + True if we use the sign of the gradient for optimization. 
        + False if we use the gradient itself
    - adv_iterations: Integer. How can interations of FSGM to perform
    '''
    
    x_adv = input_image

    print(f"y_target_index = {target_index}")
    # 1-hot encoding - For categorical cross entropy
    # TODO: Change to Sparse Categorical Cross Entropy to simplify
    y_target = tf.one_hot(target_index, preds.shape[-1])
    y_target = tf.reshape(y_target, (1, preds.shape[-1]))
    
    
    for iteration in range(adv_iterations):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            
            if add_noise:
                noise = tf.random.uniform(x_adv.shape, minval = 0, maxval=noise_max_val, dtype=tf.float32, seed=SEED)
                x_adv +=  noise
            
            # get the model prediction
            prediction = model(x_adv)
        
            x_pos = tf.identity(x_adv)
            x_neg = tf.identity(x_adv)
            
            # use perturbation of x_adv to estimate gradient
            x_pos += epsilon
            x_neg -= epsilon
            
            if add_noise:
                # the average of uniform distribution is (max - min)/2 
                x_pos += noise_max_val/2.
                x_neg -= noise_max_val/2.
            
            
            # ensure value in correct range
            x_pos = tf.clip_by_value(x_pos, -1.0, 1.0)
            x_neg = tf.clip_by_value(x_neg, -1.0, 1.0)
            
            
            pred_pos = model(x_pos)
            pred_neg = model(x_neg)
            
            pos_loss = loss_object(y_target, pred_pos)
            neg_loss = loss_object(y_target, pred_neg)
            
            
            
        print(f"Iteration {iteration}")
        print(f"Pos Loss for y_target is {pos_loss}")
        print(f"Neg Loss for y_target is {neg_loss}")
        

        # Estimate the zeroth-gradient using the 2 samples created
        gradient = (pos_loss - neg_loss) / (2*epsilon)

        # Specify what gradient we will use
        if sign_grad:
            # get the sign of the gradient
            signed_grad = tf.sign(gradient)
            # Create the adversarial image --> move in the direction of gradient sign 
            x_adv = x_adv - learning_rate * signed_grad
        else:
            x_adv = x_adv - learning_rate * gradient
            
            
        # ensure correct values (as expected by the model)
        x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
        
        
        adv_label, adv_prob, adv_preds = get_model_pred(x_adv, preproc=False)
        print(f"Iteration {iteration} - pred  = {adv_label}-{adv_prob}")
        print("="*40)
        
        # If we managed to fool the model break the loop
        if np.argmax(prediction) == target_index:
            return x_adv
        
        # if we didn't succeed just return the latest adversarial image..
        if iteration == adv_iterations-1:
            return x_adv
        






def remove_background_and_get_mask(input_image):
    # set background color to black
    bgcolor = (0,0,0,255)
    
    # Create a new session for the u2net model
    session = u2net.U2netSession(model_name="u2net", sess_opts=options)
    
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





def combine_fg_adv_bg_original(input_image, x_adv):
    def py_func(input_image, x_adv):
        # postprocess
        input_image_post = input_image[0] * 127.5 + 127.5
        x_adv_post = np.array(x_adv[0]) * 127.5 + 127.5

        x_adv_pil = Image.fromarray(x_adv_post.astype(np.uint8))
        x_adv_pil_no_bg, mask_bg, mask_fg = remove_background_and_get_mask(x_adv_pil)

        x_adv_post = np.array(x_adv_pil_no_bg)[:,:,:3]

        # combine background or the original image with foreground of attack (x_adv)
        mask_bg_array = np.array(mask_bg)
        mask_bg_array = np.stack([mask_bg_array,mask_bg_array,mask_bg_array], -1)  #(224,224,3)

        mask_fg_array = np.array(mask_fg)
        mask_fg_array = np.stack([mask_fg_array,mask_fg_array,mask_fg_array], -1)  #(224,224,3)

        x_adv_post_masked_bg = x_adv_post * mask_bg_array
        x_adv_post_masked_fg = input_image_post * mask_fg_array

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




def run_fgsm_masking_background(input_image,
                                        target_index, 
                                        model,
                                        loss_object, 
                                        epsilon = 0.05,
                                        learning_rate=0.05,
                                        sign_grad=True,
                                        adv_iterations = 50,
                                        mask_background=True):
    '''
    - input_image: tf.Tensor of shape (batch, height, width, channels)
    - target_index: Integer in [0,999] representing the y_target (we want to fool the model in predicting y_target)
    - model: pretrained tensorflow model
    - learning_rate: float for Gradient Descent
    - sign_grad: Boolent. 
        + True if we use the sign of the gradient for optimization. 
        + False if we use the gradient itself
    - adv_iterations: Integer. How can interations of FSGM to perform
    '''
    # create a copy of the input
    x_adv = tf.identity(input_image)
    
    print(f"y_target_index = {target_index}")
    y_target = tf.one_hot(target_index, preds.shape[-1])
    y_target = tf.reshape(y_target, (1, preds.shape[-1]))
    
    
    for iteration in range(adv_iterations):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            #stop gradients
            input_image = tf.stop_gradient(input_image)
            
            x_adv, mask_bg, mask_fg = combine_fg_adv_bg_original(input_image, x_adv)
            
            mask_bg = tf.stop_gradient(mask_bg)
            mask_fg = tf.stop_gradient(mask_fg)
            
            
            prediction = model(x_adv) #(1, 1000)
            loss = loss_object(y_target, prediction)
            
        print(f"Iteration {iteration} - pred  = {decode_predictions(prediction.numpy(), top=1)[0][0]}")
        print(f"Loss for y_target is {loss}")
        print("="*40)
        
    
        # If we managed to fool the model break the loop
        if np.argmax(prediction) == target_index:
            return x_adv

        # Calculate the gradient of the loss w.r.t image
        gradient = tape.gradient(loss, x_adv)
        
        if mask_background:
            # mask gradient for the background
            gradient = gradient * mask_bg
        else:
            # mask gradient for the foreground
            gradient = gradient * mask_fg
        

        if sign_grad:
            # get the sign of the gradient
            signed_grad = tf.sign(gradient)
            # Create the adversarial image --> move in the direction of gradient sign 
            x_adv = x_adv - learning_rate * signed_grad
        else:
            x_adv = x_adv - learning_rate * gradient
            
        x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)

        # if we didn't succeed just return the latest adversarial image..
        if iteration == adv_iterations-1:
            return x_adv
        







