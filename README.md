# ANN_SVM_Logisitic-regression-model
implementing regression ANN SVM on 2 datasets (1 numerical dataset + 1 dataset).
I. General Information on dataset 

I.I Numerical dataset :
We used  “Credit Card Fraud Detection”
Classes : 2 classes { Fraud , Valid } 
Fraud cases: 492
Valid Transactions: 284315
Total number of samples : 284807
Number of Training samples : 227845 ( 80% )
Testing and Validation samples : 56470 ( 20% )
Dataset link : shorturl.at/dKP02
I.II Image dataset :
We used  “Plant Pathology 2020 – FGVC7”
Classes : 4 classes { Healthy , Multiple diseases , 	Rust , Scab }
Number of samples : 1821
Size of an image : 1365 x 2048
Training samples : 1548 ( 85% )
Testing and validation samples : 273 ( 15% )
Dataset link : shorturl.at/klP29

II. Implementation details

II.I Feature extraction
Numerical dataset features : Time , amount , rest of them have hidden names
It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. 
Dimensions of features : 30
Image dataset features : we passed the whole image as feature by applying grayscale on it and resizing the image to (100x100)
II.II) Cross validation : No
II.III) Hyperparameters:
•	Learning rate: default value of the model

•	Optimizer: { logistic regression:  rmsprop , ANN : Adam } 

•	Regularization : in SVC we used 0.1

I.	Numerical : dataset was already regularized (the result of the PCA  transformation)
II.	Image : divided images on 255
•	Batch size : { Logistic regression : 16 , ANN : 16 }

•	No. of epochs : { logistic regression: 100 ,  ANN : 40}

•	Loss : { logistic regression: binary_crossentropy , ANN : sparse_categorical_entropy }

•	Callback : to keep the best weights while training and exit the training steps if the loss didn’t change for better results in 5 epochs
