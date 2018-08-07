# Keras Machine Learning Example (Identify Cats and Dogs)

[![N|Solid](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)](keras.io)



  - This is a project that was inspired on [this article](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) 
  - This project uses https://www.kaggle.com/c/dogs-vs-cats/data as a dataset
  - It was written in Python
  - Keras uses TensorFlow by Google
  - It is ~85% accurate.

# Installation

This project requires [TensorFlow](https://www.tensorflow.org/) to run.

Check if Python and Pip is installed

```sh
$ python -V  # or: python3 -V
$ pip -V     # or: pip3 -V
```

To install these packages on Ubuntu:

```sh
$ sudo apt-get install python-pip python-dev python-virtualenv   # for Python 2.7
$ sudo apt-get install python3-pip python3-dev python-virtualenv # for Python 3.n
```

We recommend using pip version 8.1 or higher. If using a release before version 8.1, upgrade pip:
```sh
$ sudo pip install -U pip
```
If not using Ubuntu and setuptools is installed, use easy_install to install pip:
```sh
$ easy_install -U pip
```
Create a directory for the virtual environment and choose a Python interpreter.
 ```sh
 $ mkdir ~/tensorflow  # somewhere to work out of
 $ cd ~/tensorflow
  # Choose one of the following Python environments for the ./venv directory:
 $ virtualenv --system-site-packages venv            # Use python default (Python 2.7)
 $ virtualenv --system-site-packages -p python3 venv # Use Python 3.n
  ```
Activate the Virtualenv environment.
Use one of these shell-specific commands to activate the virtual environment:
```sh
 $ source ~/tensorflow/venv/bin/activate      # bash, sh, ksh, or zsh
 $ source ~/tensorflow/venv/bin/activate.csh  # csh or tcsh
 $ . ~/tensorflow/venv/bin/activate.fish      # fish
When the Virtualenv is activated, the shell prompt displays as (venv) $.
```
Upgrade pip in the virtual environment.
Within the active virtual environment, upgrade pip:
```sh
(venv)$ pip install -U pip
```
You can install other Python packages within the virtual environment without affecting packages outside the virtualenv.

##### Install TensorFlow in the virtual environment.
Choose one of the available TensorFlow packages for installation:

tensorflow —Current release for CPU
tensorflow-gpu —Current release with GPU support
tf-nightly —Nightly build for CPU
tf-nightly-gpu —Nightly build with GPU support
Within an active Virtualenv environment, use pip to install the package:
```sh
$ pip install -U tensorflow
```
Use pip list to show the packages installed in the virtual environment. Validate the install and test the version:
```sh
(venv)$ python -c "import tensorflow as tf; print(tf.__version__)"
```
TensorFlow is now installed.
Use the deactivate command to stop the Python virtual environment.


##### Install Keras

Install Keras from PyPI (recommended):
```sh
$ sudo pip install keras
```

Alternatively: install Keras from the GitHub source:
First, clone Keras using git:
```sh
$ git clone https://github.com/keras-team/keras.git
```
Then, cd to the Keras folder and run the install command:
```sh
$ cd keras
$ sudo python setup.py install
```
Keras is now installed.

##### Clone this repository
Clone this repository and cd in it.
```sh
$ git clone https://github.com/ardaa/kerasml.git
$ cd kerasml
```
### Usage

If you want to train the model:
```sh
$ python train.py
```
If you want to predict using the model:
Add the directory of file that you want to predict in _predict.py_ at line 52.

```py
img_path = 'test1/1.jpg'  # Change here
# Load the image as a tensor
new_image = load_image(img_path)
```
Then run it
```sh
$ pythonw predict.py
```
### Development

Want to contribute? Great!

Fork the repository, make your changes and make a pull request.

### Todos

 - Make the image to be predicted selectable from bash
 - Improve the model to run faster.

License
----

MIT


