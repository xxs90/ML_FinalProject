# ML_FinalProject

ECE 4424 / CS 4824 Spring 2021

Professor: Lifu Huang

Author:  Guanang Su, Yijia Shi, Yuwei Wu

# Link To Google Drive Group
https://drive.google.com/drive/folders/0ACgokEX6DnrjUk9PVA

# CNN Algorithm: Running CNN_Digit_Recognition.py
1. Unzip the input file and place the input data to a local folder with a known path and change the path in the code on line 38 and 39 to the data path.
2. Required Package tensorflow, pandas, numpy, seaborn, matplotlib to run this code. The seaborn and matplotlib are required to plot the graph.
3. Using Spyder in Anaconda or using PyCharm with required package are able to run. You can change the epochs number in line 81 to reduce the running time. 
4. The output file is called cnn_submission.csv. The file records the prediction of the test data. Label represent which digit and the ids are the order of it. You can check it with the test data in test.csv.

# Logistic Regresssion Algorithm: Running LR_Digit_Recognition.py
1. Unzip the input file and place the input data to a local folder with a known path and change the path in the code on line 13 and 14 to the data path.
2. Required Package pandas, numpy, scipy matplotlib to run this code. Matplotlib are required to plot the graph.
3. Number of epoch and batch size is defined in line 145 and 146. We initialize it into 100. The running time is about 3 minutes with an accuracy about 80%. You can change by your needs.
4. The output file is called lr_submission.csv. The file records the prediction of the test data. Label represent which digit and the ids are the order of it. You can check it with the test data in test.csv.



