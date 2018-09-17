[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent" 
![Trained Agent][image1]



# Project Navigation

### Introduction

This repository contains a simple Neural Network driven Q-agent running in Unity ML Agent. This model is sometimes called "Deep Q-Network", but this is a misnomer since there is usually nothing deep in the model architecture. Typically Deep network are convolutional networks models using hundreds or thousands of layers. Our model uses a Multi Layer Perceptron with two small hidden layers of 40 nodes on each hidden layer.

The model is implemented in Python 3 using PyTorch.

Using this model, we will train an agent to navigate, collect yellow bananas and avoid blue bananas in a large, square world represented in Unity ML.  


#Goal

  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

#Rewards

A reward of +1 is provided for collecting a yellow banana.
A reward of -1 is provided for collecting a blue banana.

# State space

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions. 

# Action space

 Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Install Unity ML https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md


2. Download the Unity environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    
3. Place the file in the Navigation folder, and unzip (or decompress) the file. 

4. Install Anaconda3.

    Before you begin, you should have a non-root user with sudo privileges set up on your computer.

    The best way to install Anaconda is to download the latest Anaconda installer bash script, verify it, and then run it.

    Find the latest version of Anaconda for Python 3 at the Anaconda Downloads page.
    https://www.anaconda.com/download/#macos
    Install ANACONDA3 following instructions from the anaconda web site.
    
5. Create a virtual environment for python    
    
    In a terminal type: 
    conda create -n drlnd python=3.6
    source activate drlnd
    To stop the virtual environment once you are done, type deactivate in the terminal.
    Always start the drlnd virtual environment before starting the jupyter notebook or the python script,
    else you will get errors when running the code.
    


6. Install dependencies
    cd python
    pip install .

7. Start the notebook 
    
    cd ../
    jupyter notebook

8. Run the agent

   To start training, simply open Navigation.ipynb in Jupyter Notebook and follow the instructions given there.
   You can either train the agent: this takes about 15 minutes, or you can skip the training and watch a trained agent using the provided trained weights.



