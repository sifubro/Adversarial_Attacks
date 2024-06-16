import numpy as np
import tensorflow as tf

from helper_functions import decode_predictions

def run_fast_gradient_sign_iterative_method(input_image,
                                        target_index, 
                                        model,
                                        loss_object, 
                                        learning_rate=0.05,
                                        sign_grad=True,
                                        adv_iterations = 30,
                                        num_classes=1000):
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
    x_adv = input_image
    
    print(f"y_target_index = {target_index}")
    y_target = tf.one_hot(target_index, num_classes)
    y_target = tf.reshape(y_target, (1, num_classes))
    
    
    for iteration in range(adv_iterations):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
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