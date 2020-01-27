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
◆ Compared the accuracy of model on test data with its cross-validation accuracy and plotted some test set examples which are correctly and incorrectly.

In buildling K-means clustering algorithm on MNIST dataset, I followed these steps after shaping them as explained previously.

◆ Used MNIST test set with 10000 examples and implemented a K-means algorithm function that takes a value for the number of clusters to be found (K), a set of training examples and initial mean vector. 
◆ This function returns a n-dimensional cluster assignment(n * k one hot matrix) and a converged mean vector.
◆ At each iteration, a dot is printed as a progress indicator and once k-means objective function is minimised, the results are printed.
◆ For initializations,K.means is built with (k=10) 3 different methods : 1. Ten data points chosen uniformly at random.  2. Ten data points found using k-means++ assignment algorithm.  3. A data point from each labeled class, found by looking at test set lables).
◆ 28 * 28 images of random point from each cluster are visualized along with their cluster means.

In building a Hidden Markov Model, I followed these steps:

◆ A state mchine which mimics the "occasionally dishonest casino" is designed. This machine has two states, "Loaded" and "Fair".
◆ When in "Fair" state, it outputs a values between 1 and 6 chosen uniformly at random.
◆ When in "Loaded" state, it also outputs a value between 1 and 6, but this time the odds of emitting 1-5 are 0.1 each while the odds of emitting a 6 are 0.5.
◆ The process is modelled to start with "Fair" state, and the output of this process for 1000 steps is captured with true state of hidden variable for each step.
◆ Used forward-backward algorithm on vector of outputs, as well as true probabilities contained in transition and emission matrices to construct a MAP estimate of state distribution at each time point.
◆ Generated 2 plots of estimate state probability of a loaded die at time t, compared to actual state(which was generated earlier)
◆ One plot is estimate after performing forward pass but before computing the backward pass, and other is when entire process is complete.

In building a Linear Regression, I followed these steps:

◆ Using a crash.txt dataset , which contains measurements from a crash test dummy during NHSA automobile crash test. Column 0 measures time in milliseconds after the crash event and column 1 represents the acceleration measured by a sensor on dummy's head.
◆ Divided the data into a training and validation set of equal size - even numbered rows to training set and odd numbered rows to validate. 
◆ I built 3 models : Linear Regression with polynomial basis function, Linear Regression with radial basis function, and with Bayesian MAP estimates.
◆ Linear Regression with polynomial basis function - Fitted sets of L polynomial basis functions to training set to determine their linear weight using the normal equation. Tried L= 1,2,...20 and for each value of L, computed the maximum likelihood RMS error between actual data and model's prediction on both training and validation sets. Plotted these errors. Also, plotted both training and validation data and points with output of model function with lowest RMS on validation data.
◆ Linear Regression with radial basis function - Performed the same procedure as with polynomial fit, except used radial basis function. The function centers are evenly spaced across range of x (i.e 0-60), and the standard deviation of function is distance between one basis function mean and the next. Used L = 5,10,15,20,25 and plotted RMS on training and validation data. Also, plotted the RMS and best-fit model.
◆ To reduce the instability of ML estimates, introduced a Bayesian prior which is expressed as beta(an estimate of the inverse variance of noise in the system) and alpha (an estimate of inverse variance between the actual generative function in real world and our choice of model.). By eyeballing the noise in the data plot, it looks like a resonable estimate of standard deviation can be 20. This corresponds to beta as (1/20 * 20) i.e 0.0025. Using this as an estimate of beta, considered 100 candidate values for alpha spaced logarithically between (10 ** -8) and ( 10 ** 0). Applied this prior to the linear equation on training set and using L=50 radial basis functions. Found an appropriate value of alpha with the best performance on validation set.

In building a Logistic Regression, I followed these steps:

◆ used Iris dataset for this model. Seperated the data into features and labels and turned the label array into N * 1 array with different numbers delineating different classes into an N * K membership array (one-hot array).
◆ Added a column of ones to data to allow a bias term.
◆ Divided the features and labels in half to make proportional numbers of each class in each set.
◆ Implemented a fucntion f(w) which can be passed to scipy.optimize.minimize for gradient descent.
◆ Used linear transformations on data ( no nonlinear basis functions) which means there are four data dimensions and a bias term.
◆ Also added a gaussian prior to deal with potential degenerate behaviour. 
◆ Picked an alpha that was reasonable and implemented a objection function whose minimum is MAP estimates of weight vectors.
◆ passed this function to minimize. The best-fit posterior estimate of w obtained from before minimize function(x) is used to obtain the class posterior probalities of test set using softmax function.
◆ Classified the test set points according to highest probability produced by softmax and reported overall classification accuracy.


