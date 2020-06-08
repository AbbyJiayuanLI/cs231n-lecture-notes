# cs231n-lecture-notes
This is the collection of my notes on the course [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/) from Stanford University. 

Notes are based on my summary and lecture materials.


# 1. Introduction

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

# 2. Image Classification

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


# 3. Loss Function and Optimization

* - **Loss Function** - acquire optimal w and b
	* Data Loss L
	* Hyper-parameter  and Regularization Loss :  simpler model, less complexity
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


