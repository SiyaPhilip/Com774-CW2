import argparse
import unittest
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#import os
#train = pd.read_csv("C:\\Users\\HP\\Documents\\Com774_CW2_B00910932\\Com774-CW2\\train.csv")
#test = pd.read_csv("C:\\Users\\HP\\Documents\\Com774_CW2_B00910932\\Com774-CW2\\test.csv")
# Get the arugments we need to avoid fixing the dataset path in code
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help='Dataset for training')
parser.add_argument("--testingdata", type=str, required=True, help='Dataset for testing')
args = parser.parse_args()
mlflow.autolog()

#read data
train = pd.read_csv(args.trainingdata)
test = pd.read_csv(args.testingdata)

class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        # Set up any necessary data or objects for the tests
        # For example, load your training and testing data here
        self.x_train = train.drop(['subject','Activity'] , axis =1)
        self.y_train = train.Activity
        self.x_test = test.drop(['subject','Activity'] , axis =1)
        self.y_test = test.Activity

    def test_model_accuracy(self):
        parameters = {'max_iter': [100, 200, 500]}
        lr_classifier = LogisticRegression()
        lr_classifier_rs = RandomizedSearchCV(lr_classifier, param_distributions=parameters, cv=5, random_state=42)
        lr_classifier_rs.fit(self.x_train, self.y_train)
        y_pred = lr_classifier_rs.predict(self.x_test)

        lr_model_accuracy = accuracy_score(y_true=self.y_test, y_pred=y_pred)

        # Assert that the accuracy is within an acceptable range
        self.assertGreaterEqual(lr_model_accuracy, 0.8)

    def test_confusion_matrix_plot(self):
        parameters = {'max_iter': [100, 200, 500]}
        lr_classifier = LogisticRegression()
        lr_classifier_rs = RandomizedSearchCV(lr_classifier, param_distributions=parameters, cv=5, random_state=42)
        lr_classifier_rs.fit(self.x_train, self.y_train)
        y_pred = lr_classifier_rs.predict(self.x_test)

        cm = confusion_matrix(y_true=self.y_test, y_pred=y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(self.y_train))
        # Check that the plot does not raise any exceptions
        disp.plot(cmap='Blues')
        plt.xticks(rotation=30)

if __name__ == '__main__':
    unittest.main()

