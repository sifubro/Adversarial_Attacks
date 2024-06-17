from setuptools import setup, find_packages

setup(
    name='adversarial_attacks_white_black_box',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # Add the dependencies from your requirements.txt here
        line.strip() for line in open('requirements.txt')
    ],
    entry_points={
        'console_scripts': [
            'adversarial-attack=adversarial_attacks_white_black_box.main:main',  # Replace `main` with your main function
        ],
    },
    author='Theodoros Kasioumis',
    author_email='theodoros.kasioumis.email@example.com',
    description='Perform White-Box and Black-Box adversarial attacks on images.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/sifubro/Adversarial_Attacks',  # Replace with your project's URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Choose your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify your Python version
)