# Handwritten-Bit-Classifier
0/1 binary classifier using 2 Dense layer. 

This is simple Binary classifier(0/1) which classifies a handwritten digit in to 0 or 1. we have taken MNIST data to for training
This classifier is built with minimum no of hypermeters(very less no of nodes to train), in simple words you can train this model
with in some minutes in your CPU machine.

Techniques used:
1. Input image size was 28*28, we flattened it to 724 to fit in Dense network.
2. We have used Principal Component Analysis to reduce the input dimension from 724 to 49(we got it by visualising the Principal 
Components and we took top 49 pricipal components which covers more than 96% of the variance).
3. We have taken 2 Dense layer first having 98 nodes and second one 2 as per no of class.
4. We have taken sigmoid for intermediate layer and softmax for the end layer.
5. We have taken 1000 images per classs to train the network.
