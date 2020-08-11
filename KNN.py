import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
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
# import wind data
tech = 'onshore' # offshore  national
year = '2019'
installed_res_cap = 10000
wind_data = pd.read_csv('French Data/ninja_wind_country_FR_current-merra-2_corrected.csv', index_col=0, header=2)
# ninja_wind_country_FR_current-merra-2_corrected.csv # ninja_wind_country_FR_near-termfuture-merra-2_corrected.csv # ninja_wind_country_FR_long-termfuture-merra-2_corrected.csv
indices = [i for i in wind_data.index if i.startswith(year)]
wind_data_year_tech = wind_data.loc[indices, tech].values

# 2D K nearest neighbours

d_reduced = 3665


residual_demand_columns = [i for i in country_data.columns if i.endswith('Residual_Demand')]
X = country_data.loc[:, residual_demand_columns].values.reshape(-1, len(residual_demand_columns))
X[:, 1:] = X[:, 1:]/1.1
Y = country_data.loc[:, ['Price', 'Demand']].values.reshape(-1, 2)

scaler_Y = StandardScaler()
scaler_Y.fit(Y)
Y_scaled = scaler_Y.transform(Y)


# X_scaled = preprocessing.scale(X)
X_scaled = X
neigh = KNeighborsRegressor()
param_grid = [{'n_neighbors': [2, 5, 10, 52], 'weights': ['uniform']}]
clf = GridSearchCV(neigh, param_grid, scoring='neg_mean_squared_error', cv=100, refit=True)
clf.fit(X_scaled, Y_scaled)
Y_pred = clf.predict(X_scaled)
Y_pred_org_scale = scaler_Y.inverse_transform(Y_pred)
mse = mean_squared_error(Y[:, 0], Y_pred_org_scale[:, 0])

orders = np.argsort(X[:,0].flatten())
X_sorted = X_scaled[orders]
Y_pred_ordered = clf.predict(X_sorted)
Y_pred_ordered_org_scale = scaler_Y.inverse_transform(Y_pred_ordered)

X_reduced = np.copy(X)
X_reduced[:, 0] = X_reduced[:, 0] - installed_res_cap*wind_data_year_tech #d_reduced
Y_pred_reduced = clf.predict(X_reduced)
Y_pred_reduced_org_scale = scaler_Y.inverse_transform(Y_pred_reduced)
Y_pred_reduced_org_scale = Y_pred_reduced_org_scale[orders]

mean_reduction_in_price = (Y_pred_ordered_org_scale[:, 0].flatten() - Y_pred_reduced_org_scale[:, 0].flatten()).mean()
fig = go.Figure(data=[go.Histogram(x=Y_pred_ordered_org_scale[:, 0].flatten() - Y_pred_reduced_org_scale[:, 0].flatten())])
fig.update_layout(title="Average price reduction is  " + str(mean_reduction_in_price))
fig.write_html("figures/" + country + "histo.html")
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=X_scaled[:, 0].flatten(), y=Y[:, 0].flatten(), mode='markers', name='Original'))
fig.add_trace(go.Scatter(x=X_sorted[:, 0].flatten(), y=Y_pred_ordered_org_scale[:, 0].flatten(), mode='lines', name='Fit:' + str(clf.best_params_['n_neighbors']) + ' RMSE: ' + str(clf.best_score_)))
fig.add_trace(go.Scatter(x=X_sorted[:, 0].flatten(), y=Y_pred_reduced_org_scale[:, 0].flatten(), mode='lines', name='Fit:' + str(clf.best_params_['n_neighbors']) + ' Reduced' ))
unique, counts = np.unique((Y_pred_ordered_org_scale[:, 0] - Y_pred_reduced_org_scale[:, 0])>0, return_counts=True)
a = dict(zip(unique, counts))
fig.update_layout(
    title="Number of hours in which the price is not reduced: " + str(a[False]) + ' i.e. % ' + str(100 * a[False]/(a[False]+a[True]))
)
fig.show()
fig.write_html("figures/" + country + "_KKN_1000.html")

