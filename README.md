# simple_model-CNN

language : python 3.8

implemenation : keras

dataset : Minst

## Accuracy
![image](https://github.com/yenhao123/simple_model-CNN/blob/main/accuracy.PNG)

## method
1. beforehand (prep data、data normalization..)
2. model setting (CNN)
3. loss func setting (crossentropy)
4. training & prediction

### beforehand
1. load_data
2. normalization
3. one_hot

normalization :  make the special case influence lower 
one_hot : make the problem to a binary classification problem

## CNN intro

ideas
1. Convolution layer
2. Pooling layer

### Convolution layer

goal : feature detect

method : the input image dot filter to form a feature map

### Pooling layer

goal : feature detect

method : pick the big one in the limited matrix

## Referer

https://ithelp.ithome.com.tw/articles/10206094
