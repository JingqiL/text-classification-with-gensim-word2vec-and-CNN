# text-classification-with-gensim-word2vec-and-CNN






## A. Word Embeddings.

### 1. Max features.

Max features is the number of words that you are going to use in training and testing. I've tried to learn every word that shown in the text, the result is pretty bad because you add noise to the dataset. Most time after cleaning the dataset, there will be some words like d, aww, aaamh which originally have some numbers or punctuations in it like d'aww. The model will learn all these things which actually make it deviate from the right way. So, we gonna set a max features like 100,000 or any words shown 2 times and above. This will reduce the noise and improve the performance.

### 2. Pad Sequence.



### 3. 



## B. Test a Neural Network.

### 1. Use small batch to test it before runing on large dataset.

### 2. Failing Predictions.

Most time the dataset is extremely imbalanced, which will lead a problem that all predictions are 0 or 1. This means the neural network is not sensitive enough to detect rare cases. Some potential strategies are improving the complexity of neural networks, using oversampling to balance data (But this may lead to another problem, details will be mentioned later.)

### 3. Overfitting.

With the increase of epoches, the training performance will be better and better on the accuracy of training dataset. However, this is not true to unknown data. So, before prediction, first split the training dataset to trainset and testset, and then try a large number of epoches to see how the accuracy of testset changes. Usually, the training accuracy is increasing while the test accuracy is increasing and at some point started to falling down which indicates the overfitting of the model. 

Three methods are commonly used when overfitting happens. 

1. Add a dropout rate. Randomly choose part of nodes in the layer to calculate the linear combination.

2. Regularization. This is the same idea as Lasso and Ridge methods that we set a boundary to the sum of the |weihgts| (or the sum of the norm), which will limit the weights to be too large (Like w_ij = 2873). This will make the model to be very sensitive to a little change in the input and get a totally different prediction.

3. Don't run too many epoches.

### 4. Learning Rates.

As we know, learning rates decide how fast the neural network learning the change of the gradient. Large learning rate do lead to a faster loss decay, but when we are going to find the minimum, a big learning rate means a large learning step that we take risks on we may never get the the optimal point of the loss function. In this case, learning rate is the main reason that cause the training fail.

### 5. Batch Size.

One of the most common gradient descent method is mini-batch method. This is good not only because it save your computer from running on a very large dataset that may kill your kernel, it also lead to a much faster speed and a quite good performance on optimization. Usually the batch size is between 32-64 or sometimes 128. A large batch size sometimes will make your training performance unstable. However, for the extremely imbalanced dataset, a large batch size like 300-500 is necessary as you won't want your batch contains only one class which will stop the learning of the neural network. So, in this case, we need to use a large batch size but meanwhile, we get add more training epoches to make it learning enough and make sure the performance is good.

### 6. Filter size (CNN).

### 7. Filters.

### 8. Chanels.

### 9. Loss Function
 
The most famous two loss function known in Neural Network is Softmax and sigmoid. Softmax use the idea of standarization while sigmoid makes multi-class independent.  

When we choose loss function or define our own loss function. The first thing we need to pay attention to is that we hope our loss function doesn't have a point with derivative equals 0.

#### 1. Softmax function(Softmax cross entropy). 

Softmax function is a kind of activation function that standarized the score over all classes. The defination of the function is here. https://en.wikipedia.org/wiki/Softmax_function

Softmax cross entropy is that after applying the softmax function on your score, put it into binary cross entropy function as the loss.

#### 2. Sigmoid function(Sigmoid cross entropy).

Sigmoid function is another kind of activation function that make the original score to be rescaled to [0,1]. This activation function is applied when you have multi-class problem. (A record can have belongs to several classes instead of just one)
The definiation of the function is here. https://en.wikipedia.org/wiki/Sigmoid_function

#### 3. 
