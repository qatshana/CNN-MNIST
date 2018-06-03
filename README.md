# CNN-MNIST
Different neural networks to classify MNIST set and performance measurement for each

# 1-Layer NN with Softmax Activation Function
W=784
b=10
with 5,000 iteration adn a batch size of 100 and a learning rate of .01, we achieve a 0.901 accuracy. Please see the figure below for more details on the imporvement in accuracy vs. number of iteration
![alt text](https://github.com/qatshana/CNN-MNIST/blob/master/accuracy-vs-iteration-1-layer.png)

# 5-Layers NN with RELU Activation Function
Spec for each layer:
K=200 
L=100
M=60
N=30
Connectign layer=10

With 5000 iternation for a batch of 100 and learning rate of .1, we acheived a 0.9571 accuracy. Please see figure below more deails on the improvment in accuracy as number of iteration increases

![alt text](https://github.com/qatshana/CNN-MNIST/blob/master/CNN-5-layer%20performance.png)

# CNN with RELU Activation Function
Spec for each layer:

1) Convolutional Layer #1: Applies 32 5x5 filters (extracting 5x5-pixel subregions), with ReLU activation function
2) Pooling Layer #1: Performs max pooling with a 2x2 filter and stride of 2 (which specifies that pooled regions do not overlap)
3) Convolutional Layer #2: Applies 64 5x5 filters, with ReLU activation function
4) Pooling Layer #2: Again, performs max pooling with a 2x2 filter and stride of 2
5) Dense Layer #1: 1,024 neurons, with dropout regularization rate of 0.5 (probability of 0.5 that any given element will be dropped during training)
6) Dense Layer #2 (Logits Layer): 10 neurons, one for each digit target class (0–9).

With 5000 iternation for a batch of 100 and learning rate of .001, we acheived a 0.9901 accuracy. Please see figure below more deails on the improvment in accuracy as number of iteration increases

![alt text](https://github.com/qatshana/CNN-MNIST/blob/master/CNN-5-layer%20performance.png)
