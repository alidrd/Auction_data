import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

pio.renderers.default = 'browser'

#todo:
# residual demands minus imports/exports to neighbouring countries
# Eurostat values

# Done
# check if outliers are out properly (why percentage error increases if outliers are out, and how model choice is effected by outliers out)
# outlier removal (95% percentile)
# analysis of high error hours                                                  demand range: mid
# sum of residual demands                                                        Nope 15%



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
installed_res_cap = 34100 - 13610  #24600 - 13610
wind_data = pd.read_csv('French Data/ninja_wind_country_FR_current-merra-2_corrected.csv', index_col=0, header=2)
# ninja_wind_country_FR_current-merra-2_corrected.csv # ninja_wind_country_FR_near-termfuture-merra-2_corrected.csv # ninja_wind_country_FR_long-termfuture-merra-2_corrected.csv
indices = [i for i in wind_data.index if i.startswith(year)]
wind_data_year_tech = wind_data.loc[indices, tech].values

# K nearest neighbours

# d_reduced = 3665


# residual_demand_columns = [i for i in country_data.columns if i.endswith('Residual_Demand')]
# X = country_data.loc[:, residual_demand_columns].values.reshape(-1, len(residual_demand_columns))
# X[:, 1] = X[:, 1:].sum(axis=1)/(20)
# X = np.delete(X, [2,3,4,5,6], 1)
# Y = country_data.loc[:, ['Price', 'Demand']].values.reshape(-1, 2)

# why print(clf.scorer_) gives make_scorer(mean_squared_error, greater_is_better=False, squared=False) without squared OK with R2
# why mse not match


## outlier detection -----------------
residual_demand_columns = [i for i in country_data.columns if i.endswith('Residual_Demand')]
X_FR = country_data.loc[:, residual_demand_columns].values.reshape(-1, len(residual_demand_columns))
X_FR = np.delete(X_FR, [1, 2, 3, 4, 5, 6], 1)
Y_Pr = country_data.loc[:, ['Price']].values.reshape(-1, 1)
X_Y = np.append(X_FR, Y_Pr, 1)
# Unsupervised Outlier Detection using Local Outlier Factor (LOF)
clf = LocalOutlierFactor(n_neighbors=5, contamination=0.20) # played with n neighbours 20 50 30  cont 0.2 10:6.7 5:6.2
inliers =clf.fit_predict(X_Y)
outlier_score = clf.negative_outlier_factor_

# #isolation forest
# clf = IsolationForest(random_state=0, contamination="auto").fit(X_Y)
# inliers = clf.predict(X_Y)

# # covariance
# cov = EllipticEnvelope(random_state=0, contamination=0.2).fit(X_Y)
# inliers = cov.predict(X_Y)

print('Number of inliners are ', str(len(inliers[inliers == 1])))

#/ ---
X = country_data.loc[:, residual_demand_columns].values.reshape(-1, len(residual_demand_columns))
Y = country_data.loc[:, ['Price', 'Demand']].values.reshape(-1, 2)
X = X[inliers == 1, :]
Y = Y[inliers == 1, :]


scaler_Y = StandardScaler()
scaler_Y.fit(Y)
Y_scaled = scaler_Y.transform(Y)

scaler_X = StandardScaler()
scaler_X.fit(X)
X_scaled = scaler_X.transform(X)
FR_RD_scaler = 2
X_scaled[:, 0] = FR_RD_scaler * X_scaled[:, 0]
# print('Mean of the scaled X should be almost zero for all covariates ', X_scaled.mean(axis=0))

neigh = KNeighborsRegressor()
param_grid = [{'n_neighbors': [5, 15, 52, 168], 'weights': ['uniform']}]
# clf = GridSearchCV(neigh, param_grid, scoring='neg_mean_squared_error', cv=5, refit=True)
# clf = GridSearchCV(neigh, param_grid, scoring='neg_mean_absolute_error', cv=10, refit=True)
clf = GridSearchCV(neigh, param_grid, scoring='r2', cv=10, refit=True)
clf.fit(X_scaled, Y_scaled)
# prediction and error calculation for all data points (not just the inliers) -------------------------------
X = country_data.loc[:, residual_demand_columns].values.reshape(-1, len(residual_demand_columns))  #comment to ignore outliers in the predictions
X_scaled = scaler_X.transform(X)
X_scaled[:, 0] = FR_RD_scaler * X_scaled[:, 0]
Y = country_data.loc[:, ['Price', 'Demand']].values.reshape(-1, 2)  #comment to ignore outliers in the predictions
Y_scaled = scaler_Y.transform(Y)
# wind_data_year_tech = wind_data_year_tech[inliers == 1]          #uncomment to ignore outliers in the predictions
# ---------------------
Y_pred = clf.predict(X_scaled)
Y_pred_org_scale = scaler_Y.inverse_transform(Y_pred)
mse = mean_squared_error(Y[:, 0], Y_pred_org_scale[:, 0])
r2 = r2_score(Y[:, 0], Y_pred_org_scale[:, 0])

# Y_pred_org_scale = scaler_X.inverse_transform(Y_pred)
orders = np.argsort(X_scaled[:,0].flatten())
X_scaled_sorted = X_scaled[orders]
X_sorted_org_scale = X[orders]
Y_pred_ordered = clf.predict(X_scaled_sorted)
Y_pred_ordered_org_scale = scaler_Y.inverse_transform(Y_pred_ordered)

X_reduced = np.copy(X)
X_reduced[:, 0] = X_reduced[:, 0] - installed_res_cap*wind_data_year_tech #d_reduced
X_reduced_scaled = scaler_X.transform(X_reduced)
X_reduced_scaled[:, 0] = FR_RD_scaler * X_reduced_scaled[:, 0]
Y_pred_reduced = clf.predict(X_reduced_scaled)
Y_pred_reduced_org_scale = scaler_Y.inverse_transform(Y_pred_reduced)
Y_pred_reduced_org_scale = Y_pred_reduced_org_scale[orders]

mean_reduction_in_price = (Y_pred_ordered_org_scale[:, 0].flatten() - Y_pred_reduced_org_scale[:, 0].flatten()).mean()
fig = go.Figure(data=[go.Histogram(x=Y_pred_ordered_org_scale[:, 0].flatten() - Y_pred_reduced_org_scale[:, 0].flatten())])
fig.update_layout(title="Average price reduction is  " + str(mean_reduction_in_price) + "compared to orginal market price of " + str(Y[:,0].mean()))
fig.write_html("figures/" + country + "_histo.html")
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=X[:, 0].flatten(), y=Y[:, 0].flatten(), mode='markers', name='Original'))
fig.add_trace(go.Scatter(x=X[inliers == 1, 0].flatten(), y=Y[inliers == 1, 0].flatten(), mode='markers', name='Inliers'))   #comment to ignore outliers in the predictions
fig.add_trace(go.Scatter(x=X_sorted_org_scale[:, 0].flatten(), y=Y_pred_ordered_org_scale[:, 0].flatten(), mode='lines', name='Fit:' + str(clf.best_params_['n_neighbors']) + ' R2: ' + str(clf.best_score_)))
fig.add_trace(go.Scatter(x=X_sorted_org_scale[:, 0].flatten(), y=10*np.sign(Y_pred_ordered_org_scale[:, 0].flatten()-Y_pred_reduced_org_scale[:, 0].flatten()), mode='lines', name='Error'))
fig.add_trace(go.Scatter(x=X_sorted_org_scale[:, 0].flatten(), y=40*wind_data_year_tech[orders], mode='lines', name='New RD'))
fig.add_trace(go.Scatter(x=X_sorted_org_scale[:, 0].flatten(), y=Y_pred_reduced_org_scale[:, 0].flatten(), mode='lines', name='Fit:' + str(clf.best_params_['n_neighbors']) + ' Reduced' ))
unique, counts = np.unique((Y_pred_ordered_org_scale[:, 0] - Y_pred_reduced_org_scale[:, 0])>0, return_counts=True)
a = dict(zip(unique, counts))
fig.update_layout(
    title="Number of hours in which the price is not reduced: " + str(a[False]) + ' i.e. % ' + str(100 * a[False]/(a[False]+a[True]))
)
fig.show()
fig.write_html("figures/" + country + "_KKN_1000.html")
# demand original * price original
# demand original * price new

# fig = go.Figure(go.Histogram2d(x=X_sorted_org_scale[:, 0].flatten(), y=Y_pred_ordered_org_scale[:, 0].flatten()-Y_pred_reduced_org_scale[:, 0].flatten()))
fig = go.Figure(go.Scatter(x=X_sorted_org_scale[:, 0].flatten(), y=Y_pred_ordered_org_scale[:, 0].flatten()-Y_pred_reduced_org_scale[:, 0].flatten(), mode='markers', name='Original'))
fig.update_layout(title="Residual Demand vs. DPrice")
fig.show()

fig = go.Figure(go.Histogram2d(x=wind_data_year_tech[orders], y=Y_pred_ordered_org_scale[:, 0].flatten()-Y_pred_reduced_org_scale[:, 0].flatten()))
# fig = go.Figure(go.Scatter(x=wind_data_year_tech[orders], y=Y_pred_ordered_org_scale[:, 0].flatten()-Y_pred_reduced_org_scale[:, 0].flatten(), mode='markers', name='Original'))
fig.update_layout(title="RES vs. DPrice")
fig.show()

xx = X_sorted_org_scale.sum(axis=1)
fig = go.Figure(go.Histogram2d(x=xx, y=Y_pred_ordered_org_scale[:, 0].flatten()-Y_pred_reduced_org_scale[:, 0].flatten()))
# fig = go.Figure(go.Scatter(x=xx, y=Y_pred_ordered_org_scale[:, 0].flatten()-Y_pred_reduced_org_scale[:, 0].flatten(), mode='markers', name='Original'))
fig.update_layout(title="Sum neighbours vs. DPrice")
fig.show()

print(clf.cv_results_)
print(clf.best_estimator_)
print(clf.best_score_)
print(clf.best_params_)
print(clf.scorer_)