import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing


country = 'FR'

country_data = pd.read_csv(country + '_data_missing_data_handeled.csv', index_col=0)
import_countries = [i.split('_')[1] for i in country_data.columns if i.startswith('Imported_')]
export_countries = [i.split('_')[1] for i in country_data.columns if i.startswith('Exported_')]
neighbours = list(set(import_countries+export_countries))

for neighbour in neighbours:
    neighbour_data = pd.read_csv(neighbour + '_data_missing_data_handeled.csv', index_col=0)
    country_data[neighbour + '_Residual_Demand'] = neighbour_data.loc[:, 'Residual_Demand']

# 2D K nearest neighbours
fig = go.Figure()
d_reduced = 10000


residual_demand_columns = [i for i in country_data.columns if i.endswith('Residual_Demand')]
X = country_data.loc[:, residual_demand_columns].values.reshape(-1, len(residual_demand_columns))
Y = country_data.Price.values.reshape(-1, 1)
# X_scaled = preprocessing.scale(X)
X_scaled = X
neigh = KNeighborsRegressor()
param_grid = [{'n_neighbors': [2, 5, 10, 52], 'weights': ['uniform']}]
clf = GridSearchCV(neigh, param_grid, scoring='neg_mean_squared_error', cv=100, refit=True)
clf.fit(X_scaled, Y)
print(clf.best_params_)
print(clf.best_score_)
Y_pred = clf.predict(X_scaled)
mse = mean_squared_error(Y, Y_pred)

orders = np.argsort(X[:,0].flatten())
X_sorted = X_scaled[orders]
Y_pred_ordered = clf.predict(X_sorted)

X_reduced = np.copy(X)
X_reduced[:, 0] = X_reduced[:, 0] - d_reduced
Y_pred_reduced = clf.predict(X_reduced)
Y_pred_reduced = Y_pred_reduced[orders]

fig.add_trace(go.Scatter(x=X_scaled[:, 0].flatten(), y=Y.flatten(), mode='markers', name='Original'))
fig.add_trace(go.Scatter(x=X_sorted[:, 0].flatten(), y=Y_pred_ordered.flatten(), mode='lines', name='Fit:' + str(clf.best_params_['n_neighbors']) + ' RMSE: ' + str(clf.best_score_)))
fig.add_trace(go.Scatter(x=X_sorted[:, 0].flatten(), y=Y_pred_reduced.flatten(), mode='lines', name='Fit:' + str(clf.best_params_['n_neighbors']) + ' Reduced' ))
unique, counts = np.unique((Y_pred_ordered - Y_pred_reduced)>0, return_counts=True)
a = dict(zip(unique, counts))
fig.update_layout(
    title="Number of hours in which the price is not reduced: " + str(a[False]) + ' i.e. % ' + str(100 * a[False]/(a[False]+a[True]))
)
fig.show()
print()
fig.write_html("figures/" + country + "_KKN.html")