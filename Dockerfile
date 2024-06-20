# Use the official Python 3.9 image
FROM python:3.9

# Set the working directory
# All subsequent commands will be run in this directory
WORKDIR /app

# Copy the requirements file and install dependencies
# The following line copies the requirements.txt file from your local machine to the /app directory in the Docker container.
COPY requirements.txt .

# installs the Python packages listed in requirements.txt using pip. 
# The --no-cache-dir option prevents pip from caching the packages, which can save space
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code 
# It copies all the files and directories from your local project directory to the /app directory in the Docker container
COPY . .

# Copy templates directory to the container
# COPY templates /app/templates
# COPY static /app/static

# Expose the port the app runs on
# This line tells Docker that the container will listen on port 5000 at runtime. This is the default port that Flask uses
EXPOSE 5000

# Run the Flask app
# specifies the command to run when the container starts. It tells Docker to run the web_app.py script using Python
CMD ["python", "web_app.py"]


# Run as follows:
# cd C:\Users\SiFuBrO\Desktop\SCRIPTS!!!!!\GitHub\Adversarial_Attacks
# docker build -t adversarial_attacks_app_v2 .
# docker run -p 5000:5000 -v "$(pwd):/app" adversarial_attacks_app_v2   (linux)
# docker run -p 5000:5000 -v C://Users/SiFuBrO/Desktop/SCRIPTS!!!!!/GitHub/Adversarial_Attacks:/app adversarial_attacks_app_v2  (Windows)