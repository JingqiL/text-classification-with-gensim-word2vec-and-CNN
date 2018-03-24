# text-classification-with-gensim-word2vec-and-CNN





## Test a Neural Network

### 1. Use small batch to test it before runing on large dataset.

### 2. Failing Predictions.

Most time the dataset is extremely imbalanced, which will lead a problem that all predictions are 0 or 1. This means the neural network is not sensitive enough to detect rare cases. Some potential strategies are improving the complexity of neural networks, using oversampling to balance data (But this may lead to another problem, details will be mentioned later.)

### 3. Overfitting.

With the increase of epoches, the training performance will be better and better on the accuracy of training dataset. However, this is not true to unknown data. So, before prediction, first split the training dataset to trainset and testset, and then try a large number of epoches to see how the accuracy of testset changes. Usually, the training accuracy is increasing while the test accuracy is increasing and at some point started to falling down which indicates the overfitting of the model. 

Three methods are commonly used when overfitting happens. 

1. Add a dropout rate. Randomly choose part of nodes in the layer to calculate the linear combination.

2. Regularization. This is the same idea as Lasso and Ridge methods that we set a boundary to the sum of the |weihgts| (or the sum of the norm), which will limit the weights to be too large (Like w_ij = 2873). This will make the model to be very sensitive to a little change in the input and get a totally different prediction.

3. Don't run too many epoches.

#### 4. Learning Rates.

As we know, learning rates decide how fast the neural network learning the change of the gradient. Large learning rate do lead to a faster loss decay, but when we are going to find the minimum, a big learning rate means a large learning step that we take risks on we may never get the the optimal point of the loss function. In this case, learning rate is the main reason that cause the training fail.

### 5. Batch Size.

One of the most common gradient descent method is mini-batch method. This is good not only because it save your computer from running on a very large dataset that may kill your kernel, it also lead to a much faster speed and a quite good performance on optimization. Usually the batch size is between 32-64 or sometimes 128. A large batch size sometimes will made your training performance unstable. However, for the extremely imbalanced dataset, a large batch size like 300-500 is necessary as you won't want your batch contains only one class which will stop the learning of the neural network. So, in this case, we need to use a large batch size but meanwhile, we get add more training epoches to make it learning enough and make sure the performance is good.

### 6. Filter size (CNN).

### 7. Filters.

### 8. Chanels.

### 9. 
