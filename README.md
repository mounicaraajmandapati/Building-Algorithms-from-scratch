# Building-Algorithms-from-scratch

In this repository I added my work on building Algorithms from scratch. I built these algorithms with basic python libraries like numpy and pandas. I didnot use any advanced libraries built specificly for Machine learning like scikit-learn. The algorithms that I built from scratch include Linear Regression (with basis functions),Logistic Regression, Naive Bayes, K-Nearest Neighbor, K-means clustering, Hidden Markov model etc.


Naive Bayes on MNIST dataset:
I built a naive Bayes classifier and naive Bayes gaussian on the MNIST dataset. 
In building this classifier, I followed these steps:

◆ Downloaded  MNIST database files and loaded the contents as a list of bytes.
◆ Removed the magic numbers and bookkeeping information from the data.
◆ Turned the data into numpy array and shaped it to 60,000 rows and 784 features.
◆ Standardised the byte values from  range (0, 255) to {0,1} binary values
◆ With training images and labels, created a naive Bayes classifier. 
◆ Using this classifier(including a uniform Dirichlet prior) on test set, reported the classification accuracy.

In building naive Bayes gaussian model, I followed these steps:

◆ Used the non-thresholded 0-255 scale MNIST training set to extract 1,000 representatives of class 5 and 1,000 representatives of numbers that are not 5, chosen uniformly at random from among the other nine options.
◆ Assigned 90% of thsese selections as training set and 10% as test set.
◆ Created a naive Bayes classifier that uses gaussian kernel rather than a categorical distribution.
◆ The model has a mean parameter for each feature and each class ie for each pixel, given the category.
◆ Since, there is no enough data to compute meaningful variance for each pixel, so computed single variance for each class, using the entire category's data. This becomes the variance of entire category.
◆ Using this model, compared the True positive and false positive rates for different choices of Ʈ and plotted on an ROC curve. 

In building the K-Nearest Neighbor algorithm on MNIST dataset, I followed these steps:

◆ Used the non-thresholded 0-255 scale MNIST training set to create a training set of 200 examples of each of three classes {1,2,7}.
◆ Implemented a brute force nearest neighbor search by computing the 784 dimensional Euclidean distance.
◆ Conducted a 5-fold cross validation on training set, to pick the best candidate model of K ={1,3,5,7,9}.
◆ Created a test dataset from MNIST test data consisting of 50 examples of each of three classes {1,2,7}.
◆ Using the best Nearest neighbor model selected by model validation, classified the test data.
◆ Compared teh accuracy of model on test data with its cross-validation accuracy and plotted some test set examples which are correctly and incorrectly.
