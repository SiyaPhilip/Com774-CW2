
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import argparse
import mlflow

import warnings
warnings.filterwarnings('ignore')

# Data visualiztion Libraries
#import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.decomposition import PCA #for dimension reductionality 

from sklearn.model_selection import RandomizedSearchCV #hyperparameter and cross-validation

# machine learning model Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# for error and accurary score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay

# Get the arugments we need to avoid fixing the dataset path in code
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help='Dataset for training')
parser.add_argument("--testingdata", type=str, required=True, help='Dataset for testing')
args = parser.parse_args()
mlflow.autolog()

#read data
train_df = pd.read_csv(args.trainingdata)
test_df = pd.read_csv(args.testingdata)

train_df.head()

X_train = train_df.drop(["subject", "Activity"], axis=1)
X_test = test_df.drop(["subject", "Activity"], axis=1)
y_train = train_df.Activity
y_test = test_df.Activity

print(f"Train size: X = {X_train.shape}, y = {y_train.shape}")
print(f"Test size: X = {X_test.shape}, y = {y_test.shape}")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

activities = {label: value for label, value in enumerate(le.classes_)}

#Logistic Regression
# Generic function to run any model specified
from datetime import datetime
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

model_results= {}
def perform_model(model, x_train, y_train, x_test, y_test, cmap= plt.cm.Greens, modelname= ''):
    start_time= datetime.now()
    print('model name:', modelname)
    print('training the model...')
    model.fit(x_train, y_train)
    print('done training...')
    print('time taken:', datetime.now()- start_time)
# Predict test data
    print('predicting data...')
    start_time= datetime.now()
    y_pred= model.best_estimator_.predict(x_test)
    accuracy= metrics.accuracy_score(y_true= y_test, y_pred= y_pred)
    predtime= datetime.now() - start_time
    print('prediction time', predtime)
    print('done prediction... accuracy score = {:.2f}'.format(accuracy*100))
    print('drawing confusion matrix...')
    model_results[modelname+ '_test_score']= accuracy * 100
    model_results[modelname+ '_best_param']= model.best_params_
    model_results[modelname+ '_cv_score']= model.best_score_ * 100
    model_results[modelname+ '_prediction_time']= predtime
# draw confusion matrix
    # Assuming you have already defined y_test and y_pred
    cm = metrics.confusion_matrix(y_test,y_pred)

    plt.figure(figsize=(7, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = list(activities.values())  # Assuming you have a dictionary named 'activities'
    tick_marks = range(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.show()
    #cm = metrics.confusion_matrix(y_test, y_pred)
    #plt.figure(figsize= (7,7))
    #sns.heatmap(cm, annot= True, fmt= 'd', cmap= cmap, xticklabels= list(activities.values()), yticklabels= list(activities.values()), )
    #plt.title('Confusion Matrix')

    # normalized confusion matrix
    #ncm= cm/((cm.sum(axis=1).reshape(-1, 1)))
    #plt.figure(figsize= (7,7))
    #sns.heatmap(ncm, annot= True, fmt= '.2f', cmap= cmap, xticklabels= list(activities.values()), yticklabels= list(activities.values()), )
    #plt.title('Normalized Confusion Matrix')
    #plt.plot()
"""import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import argparse
import mlflow

import warnings
warnings.filterwarnings('ignore')

# Data visualiztion Libraries
#import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.decomposition import PCA #for dimension reductionality 

from sklearn.model_selection import RandomizedSearchCV #hyperparameter and cross-validation

# machine learning model Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# for error and accurary score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay


# Get the arugments we need to avoid fixing the dataset path in code
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help='Dataset for training')
parser.add_argument("--testingdata", type=str, required=True, help='Dataset for testing')
args = parser.parse_args()
mlflow.autolog()

#read data
train = pd.read_csv(args.trainingdata)
test = pd.read_csv(args.testingdata)
#data = pd.read_csv(args.trainingdata)
#print(data.head())

#Number of counts per activity performs by particpants
train.Activity.value_counts()

# making dataframe after splitting the columns for counting common parameter
parameter = pd.DataFrame.from_dict(Counter([col.split('-')[0].split('(')[0] for col in train.columns]),orient='index').rename(columns = {0 : 'Count'})

parameter.sort_values('Count',ascending=False)

#Model Prepration
x_train = train.drop(['subject','Activity'] , axis =1)
y_train = train.Activity

x_test = test.drop(['subject','Activity'] , axis =1)
y_test = test.Activity

print(x_test.shape , y_test.shape)
print(x_train.shape , y_train.shape)

#Preparing model with Logistic Regression
parameters = {'max_iter' : [100,200,500]}

lr_classifier = LogisticRegression()
lr_classifier_rs = RandomizedSearchCV(lr_classifier , param_distributions = parameters, 
                                      cv = 5 , random_state = 42)
lr_classifier_rs.fit(x_train , y_train)
y_pred = lr_classifier_rs.predict(x_test)

lr_model_accuracy =  accuracy_score(y_true = y_test , y_pred = y_pred)
print("Accurarcy of LR ML model :",round((lr_model_accuracy)*100 , 3))

cm = confusion_matrix(y_true = y_test , y_pred = y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm , display_labels= train.Activity.unique())
disp.plot(cmap ='Blues')
plt.xticks(rotation = 30)
plt.show()


#Preparing model with Decesion Tree
parameters = {'max_depth' : [100,200,500]}

dt_classifier = DecisionTreeClassifier()
dt_classifier_rs = RandomizedSearchCV(dt_classifier , param_distributions = parameters, 
                                      cv = 5 , random_state = 42)
dt_classifier_rs.fit(x_train , y_train)
y_pred_dt = dt_classifier_rs.predict(x_test)
dt_model_accuracy =  accuracy_score(y_true = y_test , y_pred = y_pred_dt)
print("Accurarcy of DT ML model :",round((dt_model_accuracy)*100 , 3))

cm = confusion_matrix(y_true = y_test , y_pred = y_pred_dt)
disp = ConfusionMatrixDisplay(confusion_matrix=cm , display_labels= train.Activity.unique())
disp.plot(cmap ='Blues')
plt.xticks(rotation = 30)
plt.show()

#Preparing model with SVC
parameters = {'max_iter' : [100,200,500]}

svc_classifier = SVC(kernel = 'sigmoid')
svc_classifier_rs = RandomizedSearchCV(svc_classifier , param_distributions = parameters, 
                                      cv = 5 , random_state = 42)
svc_classifier_rs.fit(x_train , y_train)
y_pred_svc = svc_classifier_rs.predict(x_test)
svc_model_accuracy =  accuracy_score(y_true = y_test , y_pred = y_pred_svc)
print("Accurarcy of SVC ML model :",round((svc_model_accuracy)*100 , 3))

cm = confusion_matrix(y_true = y_test , y_pred = y_pred_svc)
disp = ConfusionMatrixDisplay(confusion_matrix=cm , display_labels= train.Activity.unique())
disp.plot(cmap ='Blues')
plt.xticks(rotation = 30)
plt.show()"""
