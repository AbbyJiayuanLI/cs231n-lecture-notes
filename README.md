# cs231n-lecture-notes
This is the collection of my notes on the course [CS231n](http://cs231n.stanford.edu/): Convolutional Neural Networks for Visual Recognition from Stanford University. 

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
* Shortcoming of kNN:
	* Slow in time
	* Distance Matrix not informative:  L2 distance is the same if change the whole pic, therefore not good to capture perceptual difference
	* Curse of dimensionality: not dense enough
* **Second Classification: Linear Classification**
	* Parametric Model(more efficient):  f(x,w) 
		* Multiply:  f(x,w) = wx+b
		* **Insight: w as a template, and inner product shows the similarity**






问题：
解释数据集划分
猫多于狗，猫的bias term会更大

