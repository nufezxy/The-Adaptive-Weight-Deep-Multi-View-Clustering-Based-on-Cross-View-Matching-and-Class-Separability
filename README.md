# The-Adaptive-Weight-Deep-Multi-View-Clustering-Based-on-Cross-View-Matching-and-Class-Separability
This project focuses on deep multi-view clustering with adaptive feature weighting, leveraging cross-view matching and category separability. The primary goal is to develop an unsupervised learning approach that efficiently handles multi-view data, extracting relevant features from multiple perspectives to improve clustering performance. 

# System Design Document
In this project, we first construct a reconstruction loss for different views through an encoder-decoder module to learn low-dimensional feature representations. Then, two multi-layer perceptrons (MLPs) are used to extract global features and consensus representations for each view, assigning adaptive feature weights to these representations. These weights are iteratively optimized in conjunction with the clustering loss function via backpropagation.
Next, we design a cross-view matching loss function based on adaptive feature weights to minimize feature differences between the same sample in different views, achieving consistency alignment. Additionally, a margin-maximizing loss function is introduced to enhance category separability. Finally, a clustering method is employed to generate clustering results. 
![workflow](https://github.com/user-attachments/assets/edd64e03-e2cd-4216-a6f4-b48b6242f1ef)


# Specification Document
The purpose of this project is to perform deep multi-view clustering in an unsupervised manner. The input to the encoder is multi-view data, which is then processed to output initial feature representations. These outputs serve as inputs to the decoder for view reconstruction, resulting in the first loss value.
The initial feature representations are then fed into two MLPs, which output global features and specific features for each view. The specific features for each view are passed through a softmax layer to obtain distribution probabilities. These probabilities are compared with pseudo-labels generated via K-means clustering on the features, resulting in the second loss value.
Both types of feature representations and pseudo-labels are passed into the cross-view matching and margin-maximizing blocks. Through iterative optimization, adaptive feature weights are learned. These weights, along with the two feature representations, are then input into a contrastive learning loss function, yielding the third loss value. The three loss values are jointly minimized via gradient descent and backpropagation to optimize the parameters and iteratively update the global features. Finally, the global features undergo K-means clustering to obtain the clustering results.
The project is divided into two parts:
    The first part involves training for 200 epochs, focusing primarily on reconstruction.
    The second part involves setting different contrastive learning iterations based on the dataset. For example, for the BDGP dataset, 10 epochs of training will be performed. Ultimately, the system will output the best clustering result.

# User Manual
System Requirements:
Python 3.7.13
PyTorch 1.12.0
Numpy 1.21.5
Scikit-learn 0.22.2.post1
Scipy 1.7.3
Dataset:
The all datasets could be downloaded from [cloud](https://pan.baidu.com/s/18If7bx2ZOVZhyijtzycjXA). key: data
After downloading, place the dataset into the data folder of the system and preprocess it according to the dataset structure in dataloader.py. In the train.py file, set the number of training epochs for the dataset and run the script. Upon completion, the system will automatically generate a models folder and save the trained model within that folder. Finally, call the dataset in the test.py file to view the test results.

# Test Documentation
For testing experiments, I used four datasets: BDGP, Fashion, Caltech-5V, and MNIST-USPS. The training for each dataset was set as follows:
    BDGP: 10 contrastive learning epochs
    Fashion: 50 contrastive learning epochs
    Caltech-5V: 100 contrastive learning epochs
    MNIST-USPS: 50 contrastive learning epochs

# Results:
Dataset	     ACC (%)	  NMI (%)	   PUR (%)
BDGP	      99.48	       98.17	    99.48
Fashion		  97.74        95.77        97.74
Caltech-5V			
MNIST-USPS			
