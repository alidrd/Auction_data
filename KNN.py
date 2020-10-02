import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
# from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
# from sklearn.covariance import EllipticEnvelope
# from sklearn.ensemble import IsolationForest
import plotly.figure_factory as ff

pio.renderers.default = 'browser'

# todo:
# residual demands minus imports/exports to neighbouring countries
# Eurostat values

# Done
# check if outliers are out properly (why percentage error increases if outliers are out, and
# how model choice is effected by outliers out)
# outlier removal (95% percentile)
# analysis of high error hours                                                   demand range: mid
# sum of residual demands                                                        Nope 15%
# run for a given period to see if there is improvement by splitting the data to seasons  - no help
# multiple renewable technologies
# figure based on hours of the year (if error happens in a certain season) - plot and run for 4 to 8  -
# worst predictions and increased price anamolies
# including demand as input (to match hours with similar underlying demand (not residual demand) behavior )
# - triple increased price!
# checking if error of the predictor is symmetric -                              yes, good

# parameters
country = 'FR'
tech = ["onshore", "offshore"]  # tech = 'onshore' # offshore  national
year = '2019'
installed_res_cap = [24600 - 13610, 2400]  # 2023 # installed_res_cap = [34100 - 13610, 4700]   # 2028

# importing data points
country_data = pd.read_csv(country + '_data_missing_data_handeled.csv', index_col=0)
# country_data = country_data.iloc[int((5/12)*8760):int((9/12)*8760),:]   # uncomment to run for a given period only
import_countries = [i.split('_')[1] for i in country_data.columns if i.startswith('Imported_')]
export_countries = [i.split('_')[1] for i in country_data.columns if i.startswith('Exported_')]
neighbours = list(set(import_countries + export_countries))

for neighbour in neighbours:
    neighbour_data = pd.read_csv(neighbour + '_data_missing_data_handeled.csv', index_col=0)
    country_data[neighbour + '_Residual_Demand'] = neighbour_data.loc[:, 'Residual_Demand']
# residual_demand_columns = [i for i in country_data.columns if i.endswith('Residual_Demand')]
# columns_to_consider = [i for i in country_data.columns if i.endswith('Demand')]
columns_to_consider = [i for i in country_data.columns if
                       i.endswith('Residual_Demand')]  # columns_to_consider.append("Demand")
# import wind data
wind_data = pd.read_csv('French Data/ninja_wind_country_FR_current-merra-2_corrected.csv', index_col=0, header=2)
# ninja_wind_country_FR_current-merra-2_corrected.csv # ninja_wind_country_FR_near-termfuture-merra-2_corrected.csv # ninja_wind_country_FR_long-termfuture-merra-2_corrected.csv
indices = [i for i in wind_data.index if i.startswith(year)]
wind_data_year_tech = wind_data.loc[indices, tech].values
# wind_data_year_tech[:] = 0.5   # do the simulation for a given average availabitliy factor
if np.ndim(wind_data_year_tech) > 1:
    res_gen = np.sum(installed_res_cap * wind_data_year_tech, axis=1)
else:
    res_gen = installed_res_cap * wind_data_year_tech  # wind_data_year_tech = wind_data_year_tech[int((5/12)*8760):int((9/12)*8760)]   # uncomment to run for a given period only

# outlier detection
X_FR = country_data.loc[:, 'Residual_Demand'].values.reshape(-1, 1)
# X_FR = np.delete(X_FR, [1, 2, 3, 4, 5, 6], 1)
Y_Pr = country_data.loc[:, ['Price']].values.reshape(-1, 1)
X_Y = np.append(X_FR, Y_Pr, 1)
# Unsupervised Outlier Detection using Local Outlier Factor (LOF)
clf = LocalOutlierFactor(n_neighbors=5, contamination=0.20)  # played with n neighbours 20 50 30  cont 0.2 10:6.7 5:6.2
inliers = clf.fit_predict(X_Y)
outlier_score = clf.negative_outlier_factor_
# #isolation forest
# clf = IsolationForest(random_state=0, contamination="auto").fit(X_Y)
# inliers = clf.predict(X_Y)
# # covariance
# cov = EllipticEnvelope(random_state=0, contamination=0.2).fit(X_Y)
# inliers = cov.predict(X_Y)
print('Number of inliners are ', str(len(inliers[inliers == 1])))

# keeping only the inliers (all variables)
X = country_data.loc[:, columns_to_consider].values.reshape(-1, len(columns_to_consider))
Y = country_data.loc[:, ['Price', 'Demand']].values.reshape(-1, 2)
X = X[inliers == 1, :]
Y = Y[inliers == 1, :]
# scaling X and Y
scaler_Y = StandardScaler()
scaler_Y.fit(Y)
Y_scaled = scaler_Y.transform(Y)
scaler_X = StandardScaler()
scaler_X.fit(X)
X_scaled = scaler_X.transform(X)
FR_RD_scaler = 2
X_scaled[:, 0] = FR_RD_scaler * X_scaled[:, 0]
# finding KNN design
neigh = KNeighborsRegressor()
param_grid = [{'n_neighbors': [5, 15, 52, 168], 'weights': ['uniform']}]
clf = GridSearchCV(neigh, param_grid, scoring='r2', cv=10, refit=True)  # scoring='neg_mean_squared_error'
clf.fit(X_scaled, Y_scaled)

# loading all data points (not just the inliers) -------------------------------
X = country_data.loc[:, columns_to_consider].values.reshape(-1, len(
    columns_to_consider))  # comment to ignore outliers in the predictions
X_scaled = scaler_X.transform(X)
X_scaled[:, 0] = FR_RD_scaler * X_scaled[:, 0]
Y = country_data.loc[:, ['Price', 'Demand']].values.reshape(-1, 2)  # comment to ignore outliers in the predictions
Y_scaled = scaler_Y.transform(Y)
# wind_data_year_tech = wind_data_year_tech[inliers == 1]          #uncomment to ignore outliers in the predictions

# prediction and error calculation for base model
Y_pred = clf.predict(X_scaled)
Y_pred_org_scale = scaler_Y.inverse_transform(Y_pred)
mse = mean_squared_error(Y[:, 0], Y_pred_org_scale[:, 0])
r2 = r2_score(Y[:, 0], Y_pred_org_scale[:, 0])
orders = np.argsort(X_scaled[:, 0].flatten())
X_scaled_sorted = X_scaled[orders]
X_sorted_org_scale = X[orders]
Y_pred_ordered = clf.predict(X_scaled_sorted)
Y_pred_ordered_org_scale = scaler_Y.inverse_transform(Y_pred_ordered)

# prediction and error calculation for reduced residual demand
X_reduced = np.copy(X)
X_reduced[:, 0] = X_reduced[:, 0] - res_gen
X_reduced_scaled = scaler_X.transform(X_reduced)
X_reduced_scaled[:, 0] = FR_RD_scaler * X_reduced_scaled[:, 0]
Y_pred_reduced = clf.predict(X_reduced_scaled)
Y_pred_reduced_org_scale = scaler_Y.inverse_transform(Y_pred_reduced)
Y_pred_reduced_org_scale = Y_pred_reduced_org_scale[orders]

# calculating differences in Y
price_dif = Y_pred_ordered_org_scale[:, 0].flatten() - Y_pred_reduced_org_scale[:, 0].flatten()
price_dif_pos = price_dif[price_dif >= 0]
quantity_dif = Y_pred_ordered_org_scale[:, 1].flatten() - Y_pred_reduced_org_scale[:, 1].flatten()
quantity_dif_pos = quantity_dif[quantity_dif >= 0]
q_final_ordered = Y_pred_ordered_org_scale[:, 1].flatten() - quantity_dif + res_gen[orders]
price_weighted_FIP = np.average(Y_pred_ordered_org_scale[:, 0].flatten(),
                                weights=Y_pred_ordered_org_scale[:, 1].flatten())
price_weighted_FIT = np.average(Y_pred_reduced_org_scale[:, 0].flatten(), weights=q_final_ordered)
price_dif_weighted_mean = price_weighted_FIP - price_weighted_FIT
price_dif_mean = price_dif.mean()
price_dif_pos_mean = price_dif[price_dif >= 0].mean()

fig = go.Figure(
    data=[go.Histogram(x=Y_pred_ordered_org_scale[:, 0].flatten() - Y_pred_reduced_org_scale[:, 0].flatten())])
fig.update_layout(title="(Ignoring negatives) Average weighted price reduction is  " + str(
    price_dif_weighted_mean) + " (not weighted is " + str(price_dif_pos_mean) + ") vs. original wa of  " + str(
    price_weighted_FIP))
fig.write_html("figures/" + country + "_histogram.html")
fig.show()

mean_reduction_in_quantity = quantity_dif.mean()
mean_reduction_in_quantity_only_positive = quantity_dif[quantity_dif >= 0].mean()
fig = go.Figure(
    data=[go.Histogram(x=Y_pred_ordered_org_scale[:, 1].flatten() - Y_pred_reduced_org_scale[:, 1].flatten())])
fig.update_layout(title="Average quantity reduction is  " + str(int(mean_reduction_in_quantity)) +
                        " (" + str(int(mean_reduction_in_quantity_only_positive)) + " ignoring negatives)" +
                        " while average additional RES generation is " + str(int(res_gen.mean())) +
                        " making total demand increase of " + str(int(-mean_reduction_in_quantity + res_gen.mean())) +
                        " (average original Demand is " + str(Y[:, 1].mean()) + ")"
                  )
fig.show()

# other figures
fig = go.Figure()
fig.add_trace(go.Scatter(x=X[:, 0].flatten(), y=Y[:, 0].flatten(), mode='markers', name='Original'))
fig.add_trace(go.Scatter(x=X[inliers == 1, 0].flatten(), y=Y[inliers == 1, 0].flatten(), mode='markers',
                         name='Inliers'))  # comment to ignore outliers in the predictions
fig.add_trace(go.Scatter(x=X_sorted_org_scale[:, 0].flatten(), y=Y_pred_ordered_org_scale[:, 0].flatten(), mode='lines',
                         name='Fit:' + str(clf.best_params_['n_neighbors']) + ' R2: ' + str(clf.best_score_)))
fig.add_trace(go.Scatter(x=X_sorted_org_scale[:, 0].flatten(), y=10 * np.sign(
    Y_pred_ordered_org_scale[:, 0].flatten() - Y_pred_reduced_org_scale[:, 0].flatten()), mode='lines', name='Error'))
fig.add_trace(go.Scatter(x=X_sorted_org_scale[:, 0].flatten(), y=res_gen[orders] / 1000, mode='lines', name='New RD'))
fig.add_trace(go.Scatter(x=X_sorted_org_scale[:, 0].flatten(), y=Y_pred_reduced_org_scale[:, 0].flatten(), mode='lines',
                         name='Fit:' + str(clf.best_params_['n_neighbors']) + ' Reduced'))
unique, counts = np.unique((Y_pred_ordered_org_scale[:, 0] - Y_pred_reduced_org_scale[:, 0]) > 0, return_counts=True)
a = dict(zip(unique, counts))
fig.update_layout(
    title="Number of hours in which the price is not reduced: " + str(a[False]) + ' i.e. % ' + str(
        100 * a[False] / (a[False] + a[True]))
)
fig.show()
fig.write_html("figures/" + country + "_KKN_1000.html")

fig = ff.create_distplot([Y[:, 0][orders] - Y_pred_ordered_org_scale[:, 0].flatten()], ["group_labels"],
                         curve_type='normal')
fig.show()

# fig = go.Figure(go.Histogram2d(x=X_sorted_org_scale[:, 0].flatten(), y=Y_pred_ordered_org_scale[:, 0].flatten(
# )-Y_pred_reduced_org_scale[:, 0].flatten()))
fig = go.Figure(go.Scatter(x=X_sorted_org_scale[:, 0].flatten(),
                           y=Y_pred_ordered_org_scale[:, 0].flatten() - Y_pred_reduced_org_scale[:, 0].flatten(),
                           mode='markers', name='Original'))
fig.update_layout(title="Residual Demand vs. DPrice")
fig.show()

# fig = go.Figure(go.Histogram2d(x=X_sorted_org_scale[:, 0].flatten(),
# y=Y_pred_ordered_org_scale[:, 0].flatten()-Y_pred_reduced_org_scale[:, 0].flatten()))
# seasons_No = 6
# fig = go.Figure()
# for i in range(seasons_No):
#     start_hour = int((i/seasons_No)*8760)
#     end_hour = int(((i+1)/seasons_No)*8760)
#     fig.add_trace(go.Scatter(x=X[start_hour:end_hour, 0].flatten(),
#     y=Y_pred_org_scale[start_hour:end_hour, 0].flatten()-
#     scaler_Y.inverse_transform(Y_pred_reduced)[start_hour:end_hour,0], mode='markers', name=str(i)))
# fig.update_layout(title="Residual Demand vs. DPrice in seasons")
# # fig.update_yaxes(range=[-2, 0])
# # fig.update_xaxes(range=[20000, 80000])
# fig.show()

fig = go.Figure(go.Histogram2d(x=res_gen[orders],
                               y=Y_pred_ordered_org_scale[:, 0].flatten() - Y_pred_reduced_org_scale[:, 0].flatten()))
# fig = go.Figure(go.Scatter(x=wind_data_year_tech[orders],
# y=Y_pred_ordered_org_scale[:, 0].flatten()-Y_pred_reduced_org_scale[:, 0].flatten(), mode='markers', name='Original'))
fig.update_layout(title="RES vs. DPrice")
fig.show()

xx = X_sorted_org_scale.sum(axis=1)
fig = go.Figure(
    go.Histogram2d(x=xx, y=Y_pred_ordered_org_scale[:, 0].flatten() - Y_pred_reduced_org_scale[:, 0].flatten()))
# fig = go.Figure(go.Scatter(x=xx, y=Y_pred_ordered_org_scale[:, 0].flatten()-Y_pred_reduced_org_scale[:, 0].flatten(),
# mode='markers', name='Original'))
fig.update_layout(title="Sum neighbours vs. DPrice")
fig.show()

print(clf.cv_results_)
print(clf.best_estimator_)
print(clf.best_score_)
print(clf.best_params_)
print(clf.scorer_)
