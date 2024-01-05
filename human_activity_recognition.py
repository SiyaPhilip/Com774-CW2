
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
train = pd.read_csv(args.trainingdata)
test = pd.read_csv(args.testingdata)
#data = pd.read_csv(args.trainingdata)
#print(data.head())

#Number of counts per activity performs by particpants
train.Activity.value_counts()


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
print("Accurarcy of the ML model :",round((lr_model_accuracy)*100 , 3))

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
print("Accurarcy of the ML model :",round((dt_model_accuracy)*100 , 3))

cm = confusion_matrix(y_true = y_test , y_pred = y_pred_dt)
disp = ConfusionMatrixDisplay(confusion_matrix=cm , display_labels= train.Activity.unique())
disp.plot(cmap ='Blues')
plt.xticks(rotation = 30)
plt.show()

'''
# -*- coding: utf-8 -*-
"""human_activity_recognition.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uP2qDm2i5dMmsFzXGDaQSKwH0IX3cSca
"""

#pip install pandas

#pip install seaborn

#pip install matplotlib

import argparse
import cmath

from matplotlib.colors import Normalize
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import seaborn as sns
import matplotlib.pyplot as plt
import os
import mlflow
from pathlib import Path
from sklearn.manifold import TSNE
from matplotlib import cm


# Get the arugments we need to avoid fixing the dataset path in code
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help='Dataset for training')
args = parser.parse_args()
mlflow.autolog()

#read data
data = pd.read_csv(args.trainingdata)
print(data.head())

data.Activity.unique()

# Filter for only "WALKING" activity
standing_data = data[data['Activity'] == 'WALKING']

# Separate features and subject labels
X_standing = standing_data.drop(['Activity', 'subject'], axis=1)
y_subject_standing = standing_data['subject']

# Perform t-SNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
X_tsne_standing = tsne.fit_transform(X_standing)

# Create a DataFrame for the t-SNE results
tsne_standing_df = pd.DataFrame({'X': X_tsne_standing[:, 0], 'Y': X_tsne_standing[:, 1], 'Subject': y_subject_standing})
# Generate a unique color for each subject
num_subjects = y_subject_standing.nunique()

plt.figure(figsize=(12, 8))

# Assuming you have columns named 'x' and 'y' for the x and y-axis values
plt.scatter(tsne_standing_df['X'], tsne_standing_df['Y'], label='Data Points')

# Add labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('t-SNE visualization of Subjects during WALKING Activity')

# Add legend if needed
plt.legend()

# Show the plot
plt.show()


"""cmap = plt.cm.get_cmap("hsv", num_subjects)
norm = plt.Normalize(vmin=0, vmax=num_subjects-1)
palette = [cmap(norm(i)) for i in range(num_subjects)]

#cmap = cmath.get_cmap("hsv", num_subjects)
#norm = Normalize(vmin=0, vmax=num_subjects-1)s
#palette = [cmap(norm(i)) for i in range(num_subjects)]
#palette = sns.color_palette("hsv", num_subjects)

# Plotting
plt.figure(figsize=(12, 8))
plt.scatter(x='X', y='Y', hue='Subject', data=tsne_standing_df, palette=palette)
#sns.scatterplot(x='X', y='Y', hue='Subject', data=tsne_standing_df, palette=palette)
plt.title('t-SNE visualization of Subjects during WALKING Activity')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()"""

# Separate features and labels
X = data.drop('Activity', axis=1)
y = data['Activity']

# Perform t-SNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
X_tsne = tsne.fit_transform(X)

# Create a DataFrame for the t-SNE results
tsne_df = pd.DataFrame({'X': X_tsne[:, 0], 'Y': X_tsne[:, 1], 'Activity': y})

# Create a scatter plot with a specific figure size
plt.figure(figsize=(12, 8))

# Assuming you have columns named 'X' and 'Y' for the x and y-axis values
# Assuming you have a column named 'Activity' for the hue
scatter = plt.scatter(x=tsne_df['X'], y=tsne_df['Y'], c=tsne_df['Activity'], cmap='bright')

# Add labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('t-SNE Visualization of Human Activity Recognition Data')

# Add colorbar for the 'Activity' variable
plt.colorbar(scatter, label='Activity')

# Show the plot
plt.show()

"""# Plotting
plt.figure(figsize=(12, 8))
plt.scatter(x='X', y='Y', hue='Activity', data=tsne_df, palette='bright')
#sns.scatterplot(x='X', y='Y', hue='Activity', data=tsne_df, palette='bright')
plt.title('t-SNE visualization of Human Activity Recognition Data')
plt.show()"""

#Classification Model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Assuming 'Activity' is the target variable
X = data.drop('Activity', axis=1)  # Features
y = data['Activity']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix for Classification Model
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import Annotated
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
# Matplotlib heatmap
plt.matshow(cm, cmap='Blues')

# Show ticks
plt.xticks(ticks=range(len(rf_classifier.classes_)), labels=rf_classifier.classes_)
plt.yticks(ticks=range(len(rf_classifier.classes_)), labels=rf_classifier.classes_)

# Display annotations
if Annotated:
    for i in range(len(rf_classifier.classes_)):
        for j in range(len(rf_classifier.classes_)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontsize=8)

#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf_classifier.classes_, yticklabels=rf_classifier.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot feature importance
feature_importances = rf_classifier.feature_importances_

# Creating a DataFrame for visualization
feature_names = X_train.columns
importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sorting the DataFrame by importance
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(importances_df['Feature'][:10], importances_df['Importance'][:10])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importances')
plt.gca().invert_yaxis()  # To display the highest importance at the top
plt.show()

#pip install tensorflow==1.2.0 --ignore-installed

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Assuming 'Activity' is the target variable
X = data.drop('Activity', axis=1)  # Features
y = data['Activity']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Encode the labels
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)
y_train_categorical = to_categorical(y_train_encoded)
y_test_categorical = to_categorical(y_test_encoded)

# Reshape input for LSTM [samples, time steps, features]
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
# Create the LSTM model
model = Sequential()
model.add(LSTM(100, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(y_train_categorical.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_reshaped, y_train_categorical, epochs=100, batch_size=128,
                    validation_data=(X_test_reshaped, y_test_categorical))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_reshaped, y_test_categorical)
print("Accuracy: %.2f%%" % (accuracy * 100))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Create a Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train the model
gb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gb_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
# Matplotlib heatmap
plt.matshow(cm, cmap='Blues')

# Show ticks
plt.xticks(ticks=range(len(gb_classifier.classes_)), labels=gb_classifier.classes_)
plt.yticks(ticks=range(len(gb_classifier.classes_)), labels=gb_classifier.classes_)

# Display annotations
if plt.annotate:
    for i in range(len(gb_classifier.classes_)):
        for j in range(len(gb_classifier.classes_)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontsize=8)

#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=gb_classifier.classes_, yticklabels=gb_classifier.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot feature importance
feature_importances = gb_classifier.feature_importances_

# Creating a DataFrame for visualization
feature_names = X_train.columns
importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sorting the DataFrame by importance
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(importances_df['Feature'][:10], importances_df['Importance'][:10])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Gradient Boosting Feature Importances')
plt.gca().invert_yaxis()  # To display the highest importance at the top
plt.show()
'''