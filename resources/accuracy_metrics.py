#REGRESSION METRICS

'''1. R-Squared (R²)
R-Squared tells us how well the model explains the data. A value closer to 1 means the model is a good fit; a value closer to 0 means it's not.'''
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"R-Squared: {r2}")

'''2. Mean Squared Error (MSE)
MSE shows how far off the predictions are from the actual values. It squares the errors, so bigger mistakes have a larger impact.'''
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

'''3. Mean Absolute Error (MAE)
MAE measures the average of the absolute differences between predicted and actual values. It tells how far off, on average, your predictions are.'''
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.4f}")

'''4. Root Mean Squared Error (RMSE)
RMSE is just like MSE, but it takes the square root of the error, bringing it back to the same units as the target variable. It punishes larger errors more.'''
import numpy as np
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")


#CLASSIFICATION METRICS

'''1. Accuracy
Explanation: Accuracy measures how many predictions were correct out of all predictions. It's the most basic classification metric.'''
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

'''2. Precision
Explanation: Precision tells you how many of the predicted positive cases were actually positive. It's important when the cost of false positives is high.'''
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.4f}")

'''3. Recall
Explanation: Recall measures how many of the actual positive cases were correctly identified by the model. It's important when the cost of false negatives is high.'''
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.4f}")

'''4. F1-Score
Explanation: F1-Score is the harmonic mean of precision and recall. It balances precision and recall when you need a single metric to evaluate your model.'''
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)
print(f"F1-Score: {f1:.4f}")

'''5. Confusion Matrix
Explanation: A confusion matrix shows the counts of true positives, true negatives, false positives, and false negatives. It helps understand how well the model is performing.'''
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

