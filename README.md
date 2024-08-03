
Problem Statement:

The objective of this project is to develop a predictive model that can identify customers at risk of leaving the bank based on various features and details provided in the data set. The dataset includes information such as Customer ID, surname, credit score, and other relevant bank details. 
Solution:
The solution that the study offers to address the issue of customer churn is the one which is knowledge based and it hinges on the machine learning algorithms and predictive analytics, for solving the problem. Through getting data from historical customer data containing information on demographics, transaction history as well as interaction patterns, banks can spot the trends and patterns in customer behavior. The bases for building up the models based on which churn prediction can be done at the level of single customer are collected right from this stage.  The solution involves several key steps: The solution involves several key steps:  1.Data Collection and Preparation: Assemble a complete data of customers coming from various places including transactional log, survey of customer, census data and demographic information. Render the data clean and formed in order for it to get reported with greater accuracy and uniformity.  2. Feature Selection and Engineering: Emphasize on relevant attributes which impact churn rate and next do an operation on features in order to transform current ones or create new ones.  3. Model Development: Employ machine learning algorithms like logistic regression, decision trees, or ensemble methods such as random forest in the design and buildup of predictive models. Train the models by historical data techniques and score them through the correct measures.
 
1)Data Understanding Phase:

Dataset Name: Bank Customer Churn Analysis

Domain: Banking and Finance

Source of data: 
https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn- prediction?select=Churn_Modelling.csv

Atrributes:-
RowNumber ,CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary

Target Variable:- Exited 

2)Data Preprocessing(Python)
For Data preprocessing following functions were used to process the dataset:
	•	Visualization of Target Variable Distribution:
	•	The code creates a count plot to visualize the distribution of the target variable 'Exited' using seaborn's countplot() function.
	•	Each bar represents the count of customers who exited (1) and those who did not exit (0).
	•	Annotations are added to each bar to display the exact count value.
	•	This visualization helps understand the distribution of the target variable and identifies any class imbalances that may exist.
Number of Non-churned = 7963
Number of Churned = 2037

Correlation Matrix:
The relation between the attributes was found out using correlation matrix in python. The correlation matrix helps to understand relationships between different features in the dataset, helping to identify which features are strongly correlated with each other. This information can guide feature selection steps before building predictive models

Interpretation:
	•	CreditScore vs. Exited: There is a weak negative correlation (-0.027) between CreditScore and the likelihood of a customer exiting the bank. This suggests that customers with higher credit scores are slightly less likely to exit the bank.
	•	Geography vs. Exited: The correlation coefficient is -0.139, indicating a weak negative correlation between the geographic location and the likelihood of a customer exiting the bank. This suggests that customers from certain geographical locations might be slightly less likely to exit the bank compared to others.
	•	Gender vs. Exited: The correlation coefficient is -0.107, indicating a weak negative correlation between gender and the likelihood of a customer exiting the bank. This suggests that gender has a slight influence, with certain genders being slightly less likely to exit the bank.
	•	Age vs. Exited: There is a moderate positive correlation (0.285) between age and the likelihood of a customer exiting the bank. Older customers tend to be more likely to exit the bank compared to younger ones.
	•	Tenure vs. Exited: The correlation coefficient is -0.014, indicating a very weak negative correlation between tenure (the length of time the customer has been with the bank) and the likelihood of exiting the bank. This suggests that tenure has minimal impact on customer churn.
	•	Balance vs. Exited: There is a weak positive correlation (0.119) between the account balance and the likelihood of a customer exiting the bank. Customers with higher balances are slightly more likely to exit.
	•	NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary vs. Exited: These features show weak correlations with the likelihood of a customer exiting the bank. The correlations are relatively low and do not strongly indicate a significant relationship with customer churn.

Feature importance
Providing feeding and educational programs will help parents understand the importance of a nutritious diet.
Feature subset selection is a vital process of machine learning having multiple features. First of all, it solves the problem of dimensionality quandary, which means the more features there are, the more complex models become and the more computational cost is spent. As the space to explore is reduced by focused feature selection, overfitting is managed, and the model can generalise unseen data. Using Python script, we found out that: -

Interpretation
The array feature_importances represents the importance scores assigned to each feature in a machine learning model, i.e, Random Forest classifier.
	•	CreditScore: 0.1414
	•	Geography: 0.0401
	•	Gender: 0.0187
	•	Age: 0.2387
	•	Tenure: 0.0798
	•	Balance: 0.1434
	•	NumOfProducts: 0.1341
	•	HasCrCard: 0.0186
	•	IsActiveMember: 0.0401
	•	EstimatedSalary: 0.1451
Each value represents the relative importance of the corresponding feature in predicting the target variable ,i.e, ‘Exited’. Higher values indicate features that have a stronger influence on the model's predictions.
Interpretation:
	•	Age has the highest importance score (0.2387), indicating that it is the most important feature for predicting customer churn in the Random Forest model.
	•	EstimatedSalary also has relatively high importance (0.1451), suggesting that it contributes significantly to the model's predictions.
	•	Balance (0.1434) and CreditScore (0.1414) are also important features in predicting customer churn.
	•	NumOfProducts and Tenure have moderate importance scores (0.1341 and 0.0798, respectively).
	•	Geography, Gender, HasCrCard, and IsActiveMember have relatively lower importance scores compared to other features in the model.

3)Data Modelling:

Data Modelling often entails using both unsupervised and supervised learning techniques. Because we have tagged data, this dataset is intended for supervised learning.
	•	Logistic Regression:

Logistic regression is a statistical method that is used for building machine learning models where the dependent variable is dichotomous: i.e. binary. Logistic regression is used to describe data and the relationship between one dependent variable and one or more independent variables. The independent variables can be nominal, ordinal, or of interval type.

Logistic Regression Confusion Matrix (cm3):
	•	True Negatives (TN): 1571
	•	False Positives (FP): 36
	•	False Negatives (FN): 361
	•	True Positives (TP): 32

Random Forest:

Random Forest is an ensemble learning method that constructs various such decision trees as unsupervised processes and their predictions are used to obtain accuracy and robustness.

Random Forest Classifier Confusion Matrix (cm1):
	•	True Negatives (TN): 1547
	•	False Positives (FP): 60
	•	False Negatives (FN): 211
	•	True Positives (TP): 182

Interpretation:
The Random Forest classifier has the highest number of true positives (182) among the three models, indicating its effectiveness in correctly identifying customers who churn.


Decision tree:-

Decision Tree is one of the most common algorithms used by machine learning for both classification and regression problems. It use of recursive partitioning the feature space as a base for feature generations that are homogeneous for the target variable.

Decision Tree Classifier Confusion Matrix (cm2):
	•	True Negatives (TN): 1362
	•	False Positives (FP): 245
	•	False Negatives (FN): 188
	•	True Positives (TP): 205
	•	

Interpretation:
Decision Tree classifier has the lowest number of true positives and true negatives among the three models, suggesting it may not perform as well in distinguishing between churn and non-churn cases compared to the other two models.



Evaluation and Comparison of Models :

Models
Accuracy
Logistic Regression
80.15%
Random Forest
86.50%
Decision Tree
77.75%
Table- Accuracy of 3 algorithms

Based on the accuracy scores obtained from the evaluation of the three models:Based on the accuracy scores obtained from the evaluation of the three models:

1. The classifier's Random Forest had the highest of accuracy of 86.50%.
2. Logistic Regression, which is regarded, reached the accuracy of 80.15%.
3. The predictions made the decision tree reached the accuracy of 77.75%.
From reviewing Logistic Regression, Random Forest, and Decision Tree based on the dataset, it is clear that Random Forest and makes the best prediction with the highest prediction of 86.50%. This shows the highest accuracy result reaching up to 82.95% in the case of both classes while Logistic Regression (80.15%) and Decision Tree (77.75%) are considered to be less effective.


















Discussion

Using feature importance yielded that Age proved to be the factor with the highest impact in predicting the variable that is being targeted. This phenomenon puts into showcase the importance of age in decision making between whether the customer will leave or stay with the service. The ability of the model to suggest which feature most contributes to the prediction and thereby guides feature selection efforts, isolation of factors that are major drivers of the outcome accomplished and also improving model interpretability requires understanding which features contribute most to the model's predictions.

Consequently, the Random Forest model performance outshine and age as a affecting feature is important thereby it is recommended to over prioritize age-related insights by integration the Random Forest model into the decision process. Organizations can capitalize on the feature advantages of random forest, namely, having a high tolerance to overfitting, the ability to handle high dimensionality and ability to identify intricate non-linear relationships. Using this technique, they can arrive at insightful decisions based on their understanding of the reason for customer churn and accordingly develop targeted retention strategies that would enhance customer satisfaction with the aim of improving general business performance.
























ROC Curve:- 

A ROC curve is a line chart which illustrates the diagnostic capability of binary classifier systems and their performances with regard to discrimination threshold variations. It is plotted when a rope is used to suspend the body over the edge of the cliff.



Fig.13 ROC curve

ROC curve is the model usefulness criteria for the two class problems. As y-axis is TPR, it plots the TPR on y-axis against the FPR on the x-axis.

The random forest model is the one with the highest AUC (Area Under the Curve) and our AUC(AUC) is 0.86. This means that random forest model is a top-performer among other models with regard to its ability to discern between positive and negative cases. The AUC valued by decision tree algorithm is going to be 0.68, statistically speaking it is the second-highest figure, in front of logistic regression model with AUC score 0.69.

The ROC curve, in general, reveals that random forest is on top of others at distinguishing between cases that are positive and cases that are not, followed by decision tree and then logistic regression model.
