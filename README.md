# Neural_Network_Charity_Analysis
## Purpose
Use the features in the provided dataset to help create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.
 
 ## Overview
 
Over the years 34,000 organizations that have received funding from Alphabet Soup. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special consideration for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

The three technical analysis deliverables below are required.

Deliverable 1: Preprocessing Data for a Neural Network Model
Deliverable 2: Compile, Train, and Evaluate the Model
Deliverable 3: Optimize the Model
Deliverable 4: A Written Report on the Neural Network Model (README.md) 

## Resources
Dataset:  Alphabet Soup Carity dataset (Charity_data)
Software:  Python, Jupyter Notebook, Tensorflow, Alphabet Soupt Charity starter code

### Deliverable 1:  Preporcessing Data for a Neural Network Model
Using Pandas and the Scikit-Learn’s StandardScaler(), the dataset was preprocessed.

The following preprocessing steps have been performed:
The EIN and NAME columns have been dropped
![Drop_columns](https://github.com/Quinneth/Neural_Network_Charity_Analysis/blob/main/Drop_names.png)<br>

The columns with more than 10 unique values have been grouped together.
![Value Counts](https://github.com/Quinneth/Neural_Network_Charity_Analysis/blob/main/Grouping1.png)
![Replace application_counts <500](https://github.com/Quinneth/Neural_Network_Charity_Analysis/blob/main/Grouping2.png)
![Replace classification_counts <500](https://github.com/Quinneth/Neural_Network_Charity_Analysis/blob/main/Grouping4.png)

The categorical variables have been encoded using one-hot encoding.
![One_hot_encoder](https://github.com/Quinneth/Neural_Network_Charity_Analysis/blob/main/ONe_hot_encoder.png)

The preprocessed data is split into features and target arrays, then training and testing datasets.
![split_array](https://github.com/Quinneth/Neural_Network_Charity_Analysis/blob/main/Splitarray.png)

The numerical values have been standardized using the StandardScaler() module (5 pt)
![standardize](https://github.com/Quinneth/Neural_Network_Charity_Analysis/blob/main/standardize.png)

### Deliverable 2: Compile, Train, and Evaluate the Model

Using TensorFlow, the goal of this deliverable was to design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. Thought must be taken about how many inputs there are before determining the number of neurons and layers in this model. The next step was to compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

The neural network model using Tensorflow Keras contains working code that performs the following steps:
The number of layers, the number of neurons per layer, and activation function are defined, and an output layer with an activation function is created.
![Layers](https://github.com/Quinneth/Neural_Network_Charity_Analysis/blob/main/Layers.png)

Output of the model’s loss and accuracy:
![loss_accuracy](https://github.com/Quinneth/Neural_Network_Charity_Analysis/blob/main/loss%20_accuracy.png)

![Saved_results](https://github.com/Quinneth/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.h5)

## Deliverable 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model in order to achieve a target predictive accuracy higher than 75%. If you can't achieve an accuracy higher than 75%, you'll need to make at least three attempts to do so.

Optimize your model in order to achieve a target predictive accuracy higher than 75% by using any or all of the following:

### Data Preprocessing for Optimiazation Model
- The variable(s) that are considered as to be target for our model is "IS-SUCCESSFUL" coulumn, which determines the effectiveness of the fundings.
- The variable(s) considered to be features for this model are APPLICATION, TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION,  INCOME_AMT,SPECIAL_CONSIDERATIONS, STATUS, and ASK_AMT.
- The variable(s) that are neither targets nor features, and thus removed from the input data are the "EIN", "AFFILIATION","STATUS" columns as they are irrelevant and can create confusion.

### Compiling, Training, and Evaluating the Model

There are 2 hidden layers in this model. The first hidden layer has 100 neurons and the second has 50 neurons. The activation function used for both layers is "relu" and the output layer activation function is "sigmoid". These layers were selected by trial an error and they the effects can be measured by accuracy.<br>

The model was optimized to achieve 76% accuracy.<br>

Steps taken  increase model performance:
- Removal of EIN, Affiliation, and status columns, because of low relevance to model
![optimized_drop](https://github.com/Quinneth/Neural_Network_Charity_Analysis/blob/main/optimized_drop.png)
- Placed classification counts of < 1000 in Other column. 
![ooptimized other](https://github.com/Quinneth/Neural_Network_Charity_Analysis/blob/main/optimized-other.png)
- Increase hidden layers up to 4, then reduced down to the optimal 2 that provided the most accuracy
![Optimized_model](https://github.com/Quinneth/Neural_Network_Charity_Analysis/blob/main/optimized_model.png)

## Optimized Results
![Optimized_L/A](https://github.com/Quinneth/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.Optimzation.h5)
## Summary
Using a Alphabet Soup charity funding dataset, a model was created to predict successful applicants. The three features that appear most significant for for prediction are:  Name, Application_type, and Classification. Any applicant who's name appears more than 5 times, application type is T3-T8, T10, or T19, and has a classification of c1000-c3000, c12000, or c21000 has a 77% success rate. This model could be optimized further with more data points or further thought into column slection relevance. 
