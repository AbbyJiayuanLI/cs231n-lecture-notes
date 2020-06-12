# cs231n-lecture-notes
This is the collection of my notes on the course [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/) from Stanford University. 

Notes are based on my summary and lecture materials. Some points are omitted due to time limitation. 

Other useful links:
* Offical [Course Website](http://cs231n.stanford.edu/).
* [Full Syllabus](http://cs231n.stanford.edu/syllabus.html) from course website in Bilibili.
* [Course Video](https://www.bilibili.com/video/BV1nJ411z7fe?p=26) with English and Chinese caption. 
* [Translated offical notes](https://zhuanlan.zhihu.com/p/21930884) in Chinese.
* For detail notes, you may refer to [CS231n-2017-Summary](https://github.com/mbadry1/CS231n-2017-Summary) by [mbadry1](https://github.com/mbadry1). 
* For assignemnts, you may refer to [Assignemnts](https://cs231n.github.io) and answer from [CS231n-2017](https://github.com/Burton2000/CS231n-2017) by [Burton2000](https://github.com/Burton2000).


## Content
* [Demos](##demos)
* [1. Introduction](#1-introduction)
* [2. Image Classification](#2-image-classification)
* [3. Loss Function and Optimization](#3-loss-function-and-optimization)
* [4. Neural Network and Back-Propagation](#4-neural-network-and-back-propagation)
* [5. Convolutional Neural Network](#5-convolutional-neural-network)
* [6. Training Neural Network-1](#6-training-neural-network-1)
* [7. Training Neural Network-2](#6-training-neural-network-2)
* [8. Deep Learning Software](#8-deep-learning-software)
* [9. CNN Architectures](#9-cnn-architectures)
* [10. Recurrent Neural Networks](#10-recurrent-neural-networks)
* [11. Detection and Segmentation](#11-detection-and-segmentation)
* [12. Visualizing and Understanding](#12-visualizing-and-understanding)  
* [13. Generative Models](#13-generative-models)


## Demos
Here is the link for some official demos including:
* [knn](http://vision.stanford.edu/teaching/cs231n-demos/knn/) 
* [linear classification](http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/)
* [t-SNE](https://cs.stanford.edu/people/karpathy/cnnembed/)
    
## 1. Introduction

* Limited by data (labeled), computation (chip, GPU)
* ImageNet competition
* **Breakthrough** : CNN in computer vision - 2012
* **Image classification**
    * Object detection
    * Action classification
    * Image captioning 
* **CNN Development:**
    * LeCun 1998: digits recognition
* **Open Questions:**
    * Semantic Segmentation
    * Perceptual Grouping
    * Activity Recognition
    * Visual Genome: concepts and semantic relationship
    * Description of Image
    * Second/Hidden Meaning

## 2. Image Classification

* Image Classification:  assign fix category labels
* Problem:
    * Semantic Gap: between concept/semantic idea and pixel values
* Viewpoint Variation
    * Illumination
    * Deformation
    * Occlusion 遮挡
    * Background clutter (similar texture)
    * Interclass Variation
* **Data-driven Approach**
* **First Classifier: Nearest Neighbor**
    * L1 (Manhattan) Distance: depends on choice of coordinate
    * L2 (Euclidean) Distance:
    * 图片1

* **K-nearest Neighbor:** smooth out decision boundary
    * Vote weighted on distance
    * Majority Vote
* Generalization of Algorithm:  extended by choosing proper distance matrix
* Hyper-parameters: problem-dependent
    * K neighbors
	* Distance Matrix
* Choice of hyper-parameters:
    * Best Training Performance (X)
    * Split Dataset
        * Train: train and remember resultant parameters
        * Validation: compare result and choose best set of hyper-parameters
        * Test: evaluate performance
    * Cross-Validation
    * **Note: be careful with the time stamp of data. Make sure also shuffle in time
    * 图片2
* Shortcoming of kNN:
    * Slow in time
    * Distance Matrix not informative:  L2 distance is the same if change the whole pic, therefore not good to capture perceptual difference
    * Curse of dimensionality: not dense enough
* **Second Classification: Linear Classification**
    * Parametric Model(more efficient):  f(x,w)
        * Multiply:  f(x,w) = wx+b
        * **Insight: w as a template, and inner product shows the similarity**


## 3. Loss Function and Optimization

* **Loss Function** - acquire optimal w and b
	* Data Loss `L`
	* Hyper-parameter `lambda` and Regularization Loss `R(w)`:  simpler model, less complexity
		* L1 regularization: encourage sparsity (see reference) 
		* L2 regularization: derivative
		* Elastic Net = L1+L2
		* Max Norm
		* Dropout 
		* Fancier: batch normalization, stochastic depth
* **Multiclass SVM Loss**
	* Sum( Max(0,sj-syi+1) )
	* Score is only score
* **Softmax Loss** (Multinomial Logistic Regression)
	* Using exponential, normalize, the take -log
	* Score = probability 
* **Debug Note:  initialize w very small to make s approximate 0
	* Multiclass SVM Loss:  loss should be C-1 with safe gap as 1. 
	* Softmax Loss:  loss should be logC
* Optimization
	* Random Search
	* **Gradient**
		* numerical: Finite Difference (often used in gradient check in debugging)
		* Analytical: calculus
		* Gradient Descent
			* Step size/Learning Rate:  Hyper-parameter
		* Stochastic Gradient Descent (SGD): mini batch
* Image Feature:  as input of classifier, in stead of raw pixel
	* Feature Transform
		* Color histogram
		* Histogram of Orient Gradients (HoG)
		* Bag of words
		* Output 能做为下一次input？神经网络
    * 图片3

* **Extra Notes:**
    * 关于稀疏解：https://zhuanlan.zhihu.com/p/50142573
		* 从图像理解，L1用棱形逼近，解大多在坐标轴上 
		* 从导数理解，绝对值导数突变（同理，绝对值相当于逼出来一个角），或者考虑梯度下降（不太懂）
		* 从先验概率分布角度解释，（后可研究，十分巧妙）
    * 关于复杂度和regularization：
		* Minimize  L = Li (data loss) + lambda*R(w) (regularization)
			Data loss 反映accuracy，regularization反映模型复杂度，所以在minimize L时有一个accuracy和模型复杂度之间的tradeoff（因为两者线性相加）。  
    		min R(w)—>低复杂度
		* 再单从定义上看，L1: R(w)=sum(|wi|)，L2: R(w)=sum(wi^2)，minimize R(w)时会使|wi|尽可能小。
    		Min R(w) —> min |wi|
		* 所以问题就是 min |wi|为什么对应低复杂度？ 因为y=wx，如果有wi很大，那么对应的xi在变化很小的时候会引起剧烈的变化。而越简单的模型应该变化越小。


## 4. Neural Network and Back-Propagation

* **Computational Graph:  
	* calculate analytical gradient of complex functions  
	* break down calculation for large, complex matrix
* **Back Propagation:**  recursive chain rule 
	* local gradient
		* Step-by-step 
		* Local Input — Chain Rule
		* Transfer of gradient: output gradient = upstream gradient * local gradient
	* 图片4
* Patterns in BP:
	* Sigmoid Function: 
		* \sigma(x) = \frac{1}{1+e^{-x}}
		* \frac{d\sigma(x)}{dx} = (1-\sigma(x))\sigma(x)
	* Add Gate: gradient distributor 
	* Max Gate: gradient router - only max branch exerts influence and be adjusted
	* Mul Gate: gradient switcher
	* Gradient Adds
	* 图片5
* **Vectorization** — element-wise (important) 
	* Jacobian Matrix
	* 图片6
* Neural Network
	* Nonlinear Combination of linear layers 
	* Multi-discipline, integration of templates 
	* Active Function
	* 图片7
* Future Discussion:
	* Choice of Active Function
	* Network Structure/Architecture


## 5. Convolutional Neural Network

* Layer 
	* Full connected layer: stretch out
	* Convolutional Layer:  maintain spatial structure, local connectivity 
* Application 
	* Classification
	* Retrieval
	* Detection
	* Segmentation
	* Image Captioning 
* **Convolution Layer**:  WTx+b(1num)
	* Num of Filter: Multiple filter
	* Filter Type
	* Filter Size
	* Stride/Step Size
	* Zero-padding 
	* 图片8
* **Pooling Layer**
	* Hyper-parameter
		* Pooling size
		* Stride 
	* Max pooling
* **Full connected layer**


## 6. Training Neural Network-1

* **Activate Function**
	* **Sigmoid Function**: 1/(1+e^-x)
		* Good:
			* Squash to range [0,1]
		* Bad:
			* Saturated neuron kills gradients
			* Output not zero-centering, as local gradient has the same sign as x which may all be in same sign. Therefore cause zig zag update.              (so look for zero-mean data)
			* Exponential is a little bit computational expensive
			* 图片9
	* Tanh(x)
		* Good:
			* Squash to range [-1,1]  —>  zero-centering
		* Bad:
			* Still saturation problem
	* **ReLu** (Rectified Linear Unit):  f(x) = max(0, x)
		* Good:
			* Not saturated in +region
			* Computational efficient
			* Converge faster
		* Bad:
			* Not zero-centering
			* Kill gradient in -region —>  negative ReLu never update  —>  Initialization 
	* Leaky ReLu:  f(x) = max(0.1x, x)
		* Good:
			* No saturation
			* Computational efficient
			* Converge faster
			* Will not die
	* PReLu (Parametric Rectifier):   f(x) = max(ax, x)
	* ELU (Exponential Linear Unit)
		* Good:
			* Benefit of ReLu
			* More zero-center output
			* Robust to noise in -region than Leaky ReLu
		* Bad:
			* Computation requires exponential
	* Maxout:  max(w1Tx+b1, w2Tx+b2)
		* Good:
			* Generalize ReLu and Leaky ReLu
			* Linear regime, not saturate and die
		* Bad:
			* Doubles the num of parameters
	* **In Summary**:
		* Use ReLu, adjust learning rate
		* Try Leaky ReLu / Maxout / ELU
		* Try tanh, but don’t expect too much
		* Don’t use Sigmoid
* **Data Reprocessing**
	* **Zero-mean**
		* Mean image
		* per-channel mean
	* Normalization 
	* PCA
	* Whitening
* **Weight Initialization**
	* Initialized with 0
		* All neuron do same thing
	* Initialized with random std gaussian small number
		* Activation becomes 0
		* Gradient small, little update
	* Initialized with random std gaussian large number
		* Saturate and die
		* Gradient small, little update 
	* **Xavier Initialization**: scale std gaussian 
		* But collapse in ReLu
		* Extra division of 2 for the half kill in ReLu
* Batch Normalization
	* Usually after FC or Conv layer, before nonlinearity 
	* Normalization + restoring scaling and shifting
* Babysitting Learning Process
	* Preprocess 
	* Choose architecture 
	* Double check loss with and without regularization
	* Overfit small data set
	* Small regularization and find learning rate  (1e^-3, 1e^-5)
* Hyper-parameter Optimization 
	* Cross-validation
	* Fist stage: only a few epochs to get range
	* Second stage: longer time, finer search
	* Random search
	* Grid Search
	* 图片10


## 7. Training Neural Network-2

* **Fancier Optimization**
	* SGD: zig-zag when one direction sensitive the other not
		* Local Min / Saddle Point  —> gradient = 0
		* Stochastic  —>  noise 
	* **SGD+Momentum**: add momentum from current velocity
	* SGD+Nesterov Momentum: compute gradient after a step in velocity direction
	* **AdaGrad**: SGD+square gradient
		* Slower with time, which is good in convex, bad in non-convex
	* RMSProp: SGD + decay rate*square gradient
	* **Adam**: SGD + Momentum + square gradient + bias
* Learning Rate Decay:
	* Common in SGD+Momentum, not in Adam
* Second-oder Optimization:
	* Newton Method
		* Not practical as we have Hessian 
	* Quasi-Newton Method
		* Approximate inverse Hessian
	* L-BFGS (limit memory BFGS)
		* Works well in full batch, not mini batch
* **In summary, use Adam first, and try L-BFGS if full batch and less stochasticity**
* Model Ensemble 
	* Multiple Models
	* Single Model multiple shots
* egularization
	* Dropout 
		* Expectation = dropout prob * dropout output
	* Inverted Dropout
		* Test time use entire weight matrix
	* Batch Normalization
	* Data Augmentation 
		* Random Crops and Scale
		* Color Jitter
	* Drop Connect
	* Fractional Max Pooling
	* Stochastic Depth
* Transfer Learning


Adam 不能解决什么?? 
Common in SGD+Momentum, not in Adam??
L-BFGS —  stochasticity 


## 8. Deep Learning Software

* CPU vs GPU
	* CPU
		* Small
		* Faster clock
		* Memory shared with system
	* GPU (graphics processing unit)
		* Large
		* More cores - parallel 
		* Independent RAM and caching
		* Used in matrix multiplication
* **Deep Learning Framework**:
	* Reason for it:
		* Easily build bog computational graph
		* Easily compute gradient
		* Run efficiently on GPU 
	* Caffe / Caffe2
	* Theano / Tensorflow
		* Build graph
		* Compute Loss
		* Update 
			* Using dummy node with tf.group
			* Using optimizer with optimizer.minimize
		* Keras Package 
		* Static Graph
	* Torch / PyTorch
		* nn Package
		* Dynamic Graph


## 9. CNN Architectures

* **LeNet**
	* 1998 Yann LeCun
	* Digit Recognition
	* CONV-POL-CONV-POL-FC-FC
* **AlexNet**
	* 2012 krizhevsky 
	* ImageNet, first DL based
	* (CONV+POL+NORM)*2+CONV*3+POL+FC*3
	* First use ReLu and Norm Layer
	* Splitted in two GPU
* **VGG**
	* 2014 Oxford 
	* 3*(CONV*2+POOL)+2*(CONV*3/4 +POOL)+FC
	* Smaller filter, Deeper network, fewer parameters:  3 stacked 3x3 = 1 7x7
* **GoogLeNet**
	* 2014
	* (CONV+POOL) *m+ (Inception Module)*n + Linear Classifier + (auxiliary output for finer training)
	* Deeper network with computational efficiency
		* Efficient Inception module: 
			* different parallel operations on the same input
			* Filter concatenation to increase depth (only grow as pooling is inside)
			* Computation Complexity
			* Bottleneck Layer —> reduce depth and preserve dimension using 1x1 filter
		* No extra FC Layer —> less parameter
* ResNet
	* 2015
	* Very deep
	* Residual Connection - learn by +-


## 10. Recurrent Neural Networks

* Language Modeling
	* Character level
	* Word level
* **RNN**: transition matrix???
	* One to many
		* Example: Image Captioning
			* image ==> sequence of words
	* Many to One
		* Example: Sentiment Classification
			* sequence of words ==> sentiment
	* Many to many
		* Example: Machine Translation
			* seq of words in one language ==> seq of words in another language
		* Example: Video classification on frame level
* Truncated Propagation through Time (有点没懂)
* Image Captioning
	* CONV+RNN
	* With Attention: vocabulary and image location
* Visual Question Answering
* Multi-Layer RNN
* Vallina RNN Gradient Flow
	* Explode  —> gradient clipping
	* Vanish   —> more complex RNN
	* 图片11
* LSTM (Long Short-Term Memory) 1977
	* Hidden state ht
	* Cell state ct
	* Structure 
		* 图片12
* Undisrupted Gradient Flow
	* 图片13


## 11. Detection and Segmentation

* **Classification + Localization**:  single object
	* Two FC: class scores and box coordinate
* Pose Estimation
* Regression Loss: continuous 
	* L1
	* L2
* Classification: categorial 
	* Cross entropy
	* Softmax
	* SVM Loss
* **Object Detection**: multiple objects
	* Unknown nums of object
	* Sliding 
		* Hard to determine Block pos and range 
	* Region proposal
		* R-CNN
		* Fast RNN
		* Faster RNN
	* Without Proposal: regression
		* YOLO
			* Offset
			* Score 
		* SSD (single shot detection)
* **In summary, faster RNN is slower but more accurate then SSD**
* **Semantic Segmentation**: no objects, only pixels
	* Two caws within same block (cannot differentiate)
	* Sliding window
	* **Stack of CNN / Fully Convolutional**: 
		* Downsampling
		* Upsampling 
			* Nearest Neighbor
			* Bed of Nails 
			* Max unpooling
			* Transpose convolution: learnable
* **Instance Segmentation**: multiple objects, pixels
	* Mask RCNN


## 12. Visualizing and Understanding

* **Visualize CONV Layer**
	* Visualize filter of the first layer from the idea of templating
* **Visualize FC Layer**
	* Nearest Neighbor in feature space not in pixel space
	* PCA(principle component analysis):  dimension reduction to 2
	* t-SNE (t-distributed stochastic neighbor embeddings)
* **Visualize Activation Map**
* **Maximally Activating Patches**: 
	* try different patch and sort them
* **Occlusion Experiments**: 
	* mask and compare result
* **Saliency Maps**: 
	* compute gradient of class score to pixel
	* May be used in segmentation without supervision
* **Intermedia Feature** through **Guided BP**:
	* compute gradient of intermedia value to pixel
	* Only through ReLu (guided)
* **Gradient Ascent**:   generative
	* Maximize score in neuron
	* Weight w kept, change pixel value x
	* Regularization to keep image from overfitting
		* L2
		* L2 + :
			* gaussian blurring 
			* clip small value pixel to 0 
			* clip small gradient pixel to 0
	* Process:
		* Initial with zero-value Image
		* Forward and back-prop
		* Update
* **Fooling image**
	* Process 
		* Start from arbitrary image 
		* Pick arbitrary class
		* Modify image to maximize class score
	* In reality, just some noise and they appears the same in human eyes
* **DeepDream**: Amplify existing features
	* Process
		* Forward: compute chosen layer activations
		* Set chosen layer gradient to activations
		* BP: calculate gradient
		* Update image
* **Feature Inversion**
	* Match feature vectors (min their error)
* **Texture Synthesis**
	* Nearest Neighbor
	* Gram matrix: co-occurrence
* **Neural Texture Synthesis**
	* Compute Gram Matrix
* **Neural Style Transfer**: feature inversion + texture synthesis 
	* Minimize feature matching loss and gram matrix loss
	* Variations:
		* Trade-off between feature matching loss and gram matrix loss
		* Size of style image
		* Multiple styles


## 13. Generative Models

* **Supervised Learning**
	* With label
	* Learning mapping from x to y
	* Examples:
		* Classification
		* Regression
		* Object detection
		* Semantic segmentation
		* Image captioning
* **Unsupervised Learning**
	* Without label
	* Learning underlying hidden structure
	* Examples 
		* Clustering
		* PCA (dimension reduction)
		* Feature learning
		* Density estimation 
* **Generative Models**: P_data and P_model
	* Explicit density estimation
	* Implicit density estimation
	* 图片14
* **PixelRNN/CNN**: tractable density function
	* Attributes:
		* Explicit density
		* Fully visible belief nets
		* Use chain rule to decompose likelihood into 1-d distributions
	* Process:
		* Initial from corner
		* Spread with dependency modeled by RNN (LSTM) / CNN
		* Training: maximize likelihood
	* CNN faster than RNN
	* Sequential generation slow
* **Variational Auto-Encoders (VAE)**: intractable density function
	* Optimize lower bound
	* **Auto Encoder**: mapping from input to feature
		* Linear + nonlinear 
		* Deep, fully connected 
		* ReLu, CNN
		* Minimize difference between Encoder and decoder
	* Latent z: feature representations 
	* Bayesian 
		* Initialize prior with Gaussian
		* Obtain lower bound with log
		* Maximize likelihood 
	* Drawbacks: blurring and lower quality compared to GAN
* **Generative Adversarial Networks (GAN)**
	* Implicit 
	* Sample from simple distribution like random noise, and learn the transformation 
	* **Generator**: gradient ascent - max p(discriminator is wrong)
	* **Discriminator**: gradient ascent 
	* Optimize minimax function
		* Or choose proper objective function
	* Process:
		* Train discriminator 
		* Train generator
	* Convolution architecture 

 


 
