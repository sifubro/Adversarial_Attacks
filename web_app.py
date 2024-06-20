import os
import subprocess
import numpy as np
from PIL import Image

from flask import Flask, render_template, request, redirect, url_for

'''
This file contains the Flask application code that serves the web interface and handles form submissions
'''

#creates an instance of the Flask class. The __name__ argument is used to determine the root path of the application
app = Flask(__name__)

#This line defines a route for the root URL ('/') that supports both GET and POST methods
@app.route('/', methods=['GET', 'POST'])
def index():  
    '''
    This defines the index function, which is the view function for the root URL
    '''
    if request.method == 'POST': # to submit a form
        
        # retrieve the form data submitted by the user
        input_img_path = request.form['input_img_path']
        output_path = request.form['output_path']
        target_class = request.form['target_class']
        learning_rate = request.form['learning_rate']
        sign_grad = request.form['sign_grad']
        adv_iterations = request.form['adv_iterations']
        mask_background = request.form['mask_background']
        add_noise = request.form['add_noise']
        noise_max_val = request.form['noise_max_val']
        epsilon = request.form['epsilon']
        attack_method = request.form['attack_method']
        
        # Convert boolean string values to correct boolean types
        # These lines convert string values for boolean fields to actual boolean values.
        sign_grad = sign_grad.lower() == 'true'
        mask_background = mask_background.lower() == 'true'
        add_noise = add_noise.lower() == 'true'

        #This line constructs the command to run your Python script with the user-specified arguments
        command = [
            'python', 'adversarial_attacks_white_black_box/main.py',
            '--input_img_path', input_img_path,
            '--output_path', output_path,
            '--target_class', target_class,
            '--learning_rate', learning_rate,
            '--sign_grad', str(sign_grad),
            '--adv_iterations', adv_iterations,
            '--mask_background', str(mask_background),
            '--add_noise', str(add_noise),
            '--noise_max_val', noise_max_val,
            '--epsilon', epsilon,
            '--attack_method', attack_method
        ]

        # This line runs the constructed command using the subprocess module
        subprocess.run(command)
        # This line redirects the user back to the root URL after the script runs
        # return redirect(url_for('index'))
        # Redirect to results page:
        return redirect(url_for('results', input_img_path=input_img_path, output_path=output_path))
    
    # If the request method is GET, this line renders the HTML form for the user to fill out
    return render_template('index.html')

@app.route('/results')
def results():

    return render_template('results.html')


# checks if the script is being run directly (not imported as a module)
if __name__ == '__main__':
    # This line runs the Flask app on host 0.0.0.0 (making it accessible externally) and port 5000.
    app.run(host='0.0.0.0', port=5000)
