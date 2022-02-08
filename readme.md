#  Neural Network Hyperparameters
• Describe the methods you used for your experiment. This should include a discussion of the dataset (e.g., source? number of examples?) and what parameters were used to train all the models.
– The Data set is sklearn load digits dataset. The Dataset is 2d matrix datasets of pixel bitmaps of handwritten numbers in the range [0-9]. Each datapoint is a 8x8 image of a digit. The data set contains images of hand written digits, which is 10 classes where each class refers to a digit. 32 × 32 bitmaps are divided blocks of 4 × 4 and number of ON pixels are counted in each block. This generates a matrix of [8 × 8] where each element is in range [0....16].
∗ Classes 10
∗ Samples per class 180 ∗ Samples total 1797
∗ Dimensions 64
∗ Features - integers 0-16
– The input for the Multilayer Perceptron will be 64 values (since there are 8 × 8 matrix fields) and the output will be 10 (the digits 0-9). The input data used to train the model is the count of number of ON pixels in a 4 × 4 pixel bitmap. The pereceptron model is fit on the Training data set and the Target dataset
– The data is split into 80-20 train test sets, and after splitting the X train data is 1437 × 64 X test.shape is 360 × 64)
– In this experiment we have used MLPClassifier of sklearn.neuralNetwork package. Multi-layer Perceptron classifier uses stochastic gradient descent model to optimize the loss function. The parameters that we used to perform this experiment are:
∗ hidden layer sizes : The ith element represents the number of neurons in the ith hidden layer. For the purpose of this experiment we have choosen max two Hidden layers with 100 and 200 neurons per layers respectively.
∗ activation : We have used ReLu and tanh as activation
∗ max iter : Maximum number of iterations set so that the model’s loss functions can converge.
• Report your results for every tested model. Introducing hidden nodes with combinations of activation function, number of neuron, and hidden layers.
– single hidden layer with tanh activation function
∗ perceptron1 with hiddenlayers = 1, number of neurons = 100, Accuracy on Test data 0.977777 ∗ perceptron2 with hiddenlayers = 1, number of neurons = 200 Accuracy on Test data 0.977777
– Two hidden layers with tanh activation function
∗ perceptron3 with hiddenlayers = 2, number of neurons = [100, 200] Accuracy on Test data 0.983333
∗ perceptron4 with hiddenlayers = 2, number of neurons = [200, 100] Accuracy on Test data 0.983333
– single hidden layer with ReLu activation function
∗ perceptron5 with hiddenlayers = 1, number of neurons = [100,] Accuracy on Test data 0.983333
∗ perceptron6 with hiddenlayers = 1, number of neurons = [200,] Accuracy on Test data 0.9888888888888889
– Two hidden layers with ReLu activation function
2.2
•
•
∗ perceptron7 with hiddenlayers = 2, number of neurons = [100, 200] Accuracy on Test data 0.9861111111111112
∗ perceptron8 with hiddenlayers = 2, number of neurons = [200, 100] Accuracy on Test data 0.9861111111111112
The confusion matrix against predicted and actual values for all the classes [0-9], is plotted for each model and attached in the code part.
Discuss your analysis of what general trends emerge from your results. For example, did a certain number of hidden layers, number of neurons per layer, or activation functions lead to consistently better results. If so, why do you think this occurs? You also could analyze examine what, if any, insights are gained by looking at both the different evaluation approaches (i.e., accuracy and confusion matrix).
– One straightforward observation that one can make is from the experiment is that if we fix the max iterations i.e the number of epochs of running the model, than the accuracy obtained when using two hidden layers is higher than when we use only one hidden layer. A very obvious explanation to this an be that, when using more hidden layers, the loss functions can converge in lesser number of iterations than compared when using few hidden layers. The weight updates at each layer is more close to convergent weight in the case of multiple hidden layers.
– When using tanh activation function with only one hidden layer, on increasing the number of neurons at the hidden layer, there was no visible change in the Accuracy, while for ReLu there was significant change in the accurancy score on doubling the number of neurons.
– When using two hiddenlayers in the experiment the ordering of number of neuron per hidden layer didn’t effect the accuracy score of the model for both ReLu and tanh activation functions.
– *Comparing the activation functions we can observed that for the same number of epochs ReLu was able to achieve better accurancy score than tanh for single and double hiddenlayer models. This can be greatly credited to the fact that ReLu doesnot have a saturation in ts gradient which helps in accelerating the converges of stochastic gradient descent in lesser number of epochs than tanh.
– On increasing the number of hidden layers much more than the what is sufficient can cause the accuracy in the test set to decrease as the model will overfit the training set, and the model won’t be able to generalize new unseen data.
Impact of Training Duration and Training Data
Describe the methods you used for your experiment. This should include a discussion of the dataset and the parameters used to train all the models.
– The Data set is sklearn load digits dataset. The Dataset is 2d matrix datasets of pixel bitmaps of handwritten numbers in the range [0-9]. Each datapoint is a 8x8 image of a digit. The data set contains images of hand written digits, which is 10 classes where each class refers to a digit. 32 × 32 bitmaps are divided blocks of 4 × 4 and number of ON pixels are counted in each block. This generates a matrix of [8 × 8] where each element is in range [0....16].
∗ Classes 10
∗ Samples per class 180 ∗ Samples total 1797
∗ Dimensions 64
∗ Features - integers 0-16
– The input for the Multilayer Perceptron will be 64 values (since there are 8 × 8 matrix fields) and the output will be 10 (the digits 0-9). The input data used to train the model is the count of number of ON pixels in a 4 × 4 pixel bitmap. The perceptron model is fit on the Training data set and the Target dataset

• Show the plot that visualizes the performance for each of the four approaches.
• Discuss your analysis of what general trends emerge from your results. For example, what is the influence of the amount of training data and the training duration?
– For the training where we chose 100/0 split of the train and test data, we can see that the accuracy of the model is always a constant 100%. This can be considered an example of overfitting a model. Since we have no new data to visualize, the model has already seen all the data that its being tested on.
– For the data set with Training data to be 75% , we can observe the maximum accuracy in comparison to 25 and 50.
– The number of epochs needed for the loss functino to convergence is inversely proportional to train data size.
– The model with training data of 25% seems to perform poorly over the test data set in comparison to other models and can be considered as a case of underfitting model.
2.3 References
• scikit-learn.org/stable/modules/generated/sklearn.datasets.load digits
• https://datascience.stackexchange.com/
