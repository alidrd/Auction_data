import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sklearn.model_selection
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


data = np.random.uniform(0, 1,(1000, 1))
noise = np.random.normal(size=(1000,))
X = data[:,:1]
y = 10.0*(data[:,0]) + noise

plt.plot(X[:,0], y, '.')
plt.xlabel('X (input variable)')
plt.ylabel('y (target variable)')
plt.title("Data")

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.4, random_state=2019)
#%% full trees
rf = RandomForestRegressor(n_estimators=50)
rf.fit(X_train, y_train)
y_train_predicted = rf.predict(X_train)
y_test_predicted_full_trees = rf.predict(X_test)
mse_train = mean_squared_error(y_train, y_train_predicted)
mse_test = mean_squared_error(y_test, y_test_predicted_full_trees)
print("RF with full trees, Train MSE: {} Test MSE: {}".format(mse_train, mse_test))

#%% pruned trees
rf = RandomForestRegressor(n_estimators=50, min_samples_leaf=25)
rf.fit(X_train, y_train)
y_train_predicted = rf.predict(X_train)
y_test_predicted_pruned_trees = rf.predict(X_test)
mse_train = mean_squared_error(y_train, y_train_predicted)
mse_test = mean_squared_error(y_test, y_test_predicted_pruned_trees)
print("RF with pruned tree, Train MSE: {} Test MSE: {}".format(mse_train, mse_test))

#%% grid search
rfm = RandomForestRegressor(bootstrap=True, random_state=42, oob_score=False, criterion='mae', n_jobs=8)
param_grid = [{'n_estimators': [10, 20, 50], 'min_samples_leaf': [10, 20, 50], 'max_depth': [10, 50]}]
rf = GridSearchCV(rfm, param_grid, scoring='r2', cv=10, refit=True)  # scoring='neg_mean_squared_error'
rf.fit(X, y)
y_train_predicted = rf.predict(X_train)
y_test_predicted_pruned_trees = rf.predict(X_test)
mse_train = mean_squared_error(y_train, y_train_predicted)
mse_test = mean_squared_error(y_test, y_test_predicted_pruned_trees)
print("RF with grid search, Train MSE: {} Test MSE: {}".format(mse_train, mse_test))
print(rf.best_estimator_)
print(rf.best_score_)

#%%
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(X_test[:,0], y_test_predicted_full_trees,'.')
plt.title("Random Forest with full trees")
plt.xlabel("X (input variable)")
plt.ylabel("Random Forest prediction")

plt.subplot(122)
plt.plot(X_test[:,0], y_test_predicted_pruned_trees,'.')
plt.title("Random Forest with pruned trees")
plt.xlabel("X (input variable)")
plt.ylabel("Random Forest prediction")
plt.show()



#%%
# rf = RandomForestRegressor(n_estimators=1)
# trees, train_loss, test_loss = [], [], []
# for iter in range(50):
#   rf.fit(X_train, y_train)
#   y_train_predicted = rf.predict(X_train)
#   y_test_predicted = rf.predict(X_test)
#   mse_train = mean_squared_error(y_train, y_train_predicted)
#   mse_test = mean_squared_error(y_test, y_test_predicted)
#   print("Iteration: {} Train mse: {} Test mse: {}".format(iter, mse_train, mse_test))
#   trees += [rf.n_estimators]
#   train_loss += [mse_train]
#   test_loss += [mse_test]
#   rf.n_estimators += 1
# plt.figure(figsize=(8,6))
# plt.plot(trees, train_loss, color="blue", label="MSE on Train data")
# plt.plot(trees, test_loss, color="red", label="MSE on Test data")
# plt.xlabel("# of trees")
# plt.ylabel("Mean Squared Error")
# plt.legend()