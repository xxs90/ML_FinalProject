# ML_FinalProject

ECE 4424 / CS 4824 Spring 2021
Professor: Lifu Huang

# Link To Google Drive Group
https://drive.google.com/drive/folders/0ACgokEX6DnrjUk9PVA

# Project Description
Overview

The final project for the Machine Learning course will be a mini-research project done by groups of up to 3 people. The overall goal of such a project is to allow students to gain basic skills of using machine learning algorithms to solve practical problems.

You can choose one of the three topics we suggested, or choose any other topics that you are interested in. You need to implement at least two machine learning algorithms to solve the problem, report their performance and analyze their results, e.g., what works and what not, what category of errors are still remaining, what advantages and disadvantages you observed, how did you learn and tune the parameters, etc.

Several things you need to pay attention to:

    When you choose the topic, you need to make sure there exists manually annotated data available for you to train the models and evaluate the performances.
    You are not allowed to directly copy code from online or others. You are not allowed to use machine learning packages, e.g., Scikit-Learn. You can use Pytorch or Tensorflow if you use Convolutional Neural Networks, Recurrent Neural Networks, etc.
    Each team will be assigned a TA to help. You are highly encouraged to discuss with your TA when you choose the topic to work on.

Deliverables

1. The first deliverable will be the programs you have implemented:

    the source code as well as the data you used to train and test the models
    the optimized parameters/models you have learned
    a README file to explain how to step-by-step train and test the models

2. The second deliverable will be a final report, which should be at least two pages long in a reasonable format. You should use at least 10-point font in a standard typeface such as Times New Roman, Arial, or Helvetica. It should discuss:

    what the problem is and why it's important
    what algorithms you used to solve the problem and details of how the algorithms work
    the experimental results, including detailed performance, e.g., Precision, Recall, F-score, detailed discussion and analysis of the results, e.g., remaining errors, advantages and disadvantages of each algorithm you observed, how to further improve them, etc. 
    describe the role of each team member within the project

3. The final deliverable is a final project presentation, which should cover all the content you discuss in the final report. 
Project Ideas

Topic:

Handwritten Digit Recognition: given an image, predict the corresponding digit from 0 - 9.

        Dataset: you can access the train/test datasets from Canvas (Files -> Final Project Datasets -> digit-recognition.zip). Each data instance has the following attributes:

        label: a label that marks the digit of the corresponding image
        pixel0 - pixel783: each input image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.


4. Other interesting problems can be found: 

    Kaggle: https://www.kaggle.com/
    SemEval 2020: https://alt.qcri.org/semeval2020/index.php?id=tasks (Links to an external site.)
    Conference workshops, e.g., https://2020.emnlp.org/workshops (Links to an external site.)


