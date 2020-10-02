import pandas as pd
import numpy as np
import plotly.io as pio
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from sklearn.neighbors import LocalOutlierFactor


pio.renderers.default = 'browser'
#%% parameters
country = 'FR'
tech = ["onshore"]  # tech = 'onshore' # offshore  national tech = ["onshore", "offshore"]
year = '2019'
# installed_res_cap = [24600 - 13610, 2400]  # 2023 # installed_res_cap = [34100 - 13610, 4700]   # 2028
#%% importing data points
#todo: add temperature
country_data = pd.read_csv(country + '_data_missing_data_handeled.csv', index_col=0)
country_data["hour_in_day"] = (country_data["hour"]-1) % 24
res_columns = ['Wind Onshore', 'Solar', 'Waste', 'Hydro Run-of-river and poundage', 'Wind Offshore', 'Marine', 'Other renewable']
country_data["res"] = country_data[res_columns].sum(axis=1)
import_countries = [i.split('_')[1] for i in country_data.columns if i.startswith('Imported_')]
export_countries = [i.split('_')[1] for i in country_data.columns if i.startswith('Exported_')]
neighbours = list(set(import_countries + export_countries))

for neighbour in neighbours:
    neighbour_data = pd.read_csv(neighbour + '_data_missing_data_handeled.csv', index_col=0)
    country_data[neighbour + '_Residual_Demand'] = neighbour_data.loc[:, 'Residual_Demand']
#%% outlier detection
X_FR = country_data.loc[:, 'Residual_Demand'].values.reshape(-1, 1)
# X_FR = np.delete(X_FR, [1, 2, 3, 4, 5, 6], 1)
Y_Pr = country_data.loc[:, ['Price']].values.reshape(-1, 1)
X_Y = np.append(X_FR, Y_Pr, 1)
# Unsupervised Outlier Detection using Local Outlier Factor (LOF)
clf = LocalOutlierFactor(n_neighbors=5, contamination=0.20)  # played with n neighbours 20 50 30  cont 0.2 10:6.7 5:6.2
inliers = clf.fit_predict(X_Y)
outlier_score = clf.negative_outlier_factor_
print('Number of inliners are ', str(len(inliers[inliers == 1])))
# keeping only the inliers (all variables)
#%% defining Xs and Ys
columns_to_consider = [i for i in country_data.columns if i.endswith('_Residual_Demand')]
columns_to_consider = columns_to_consider + ["hour", "hour_in_day", 'res']
X = country_data.loc[:, columns_to_consider].values.reshape(-1, len(columns_to_consider))
Y = country_data.loc[:, ['Price', 'Demand']].values.reshape(-1, 2) #,  2
## X = X[inliers == 1, :]
## Y = Y[inliers == 1, :]
# rf = RandomForestRegressor(n_estimators=50, bootstrap=True, random_state=42, oob_score=True, criterion='mae', n_jobs=10)
# print('here2')
# rf.fit(X, Y)
# predictions = rf.predict(X)
# print('here4')
# #%% reporting accuracy
# error_1 = abs(Y[:, 0]-predictions[:, 0])
# error_2 = abs(Y[:, 1]-predictions[:, 1])
# print("MAE for P is ", round(np.mean(error_1), 2), "EUR/MWh")
# print("MAE for D is ", round(np.mean(error_2), 2), "MWH")
# mape_1 = 100 * (error_1 / Y[:, 0])
# mape_no_inf = np.ma.masked_invalid(mape_1)
# mape_2 = 100 * (error_2 / Y[:, 1])
# print('Accuracy P:', round(100-np.mean(mape_no_inf), 2), '%.')
# print('Accuracy D:', round(100-np.mean(mape_2), 2), '%.')
# #%% wind analysis
# A = 13610  #todo find max A
# wind_data = pd.read_csv('French Data/ninja_wind_country_FR_current-merra-2_corrected.csv', index_col=0, header=2)
# # ninja_wind_country_FR_current-merra-2_corrected.csv # ninja_wind_country_FR_near-termfuture-merra-2_corrected.csv # ninja_wind_country_FR_long-termfuture-merra-2_corrected.csv
# indices = [i for i in wind_data.index if i.startswith(year)]
# wind_data_year_tech = wind_data.loc[indices, tech].values
# # wind_data_year_tech[:] = 0.5   # do the simulation for a given average availabitliy factor
# if np.ndim(wind_data_year_tech) > 1:
#     res_gen_to_remove = np.sum(A * wind_data_year_tech, axis=1)
# else:
#     res_gen_to_remove = A * wind_data_year_tech  # wind_data_year_tech = wind_data_year_tech[int((5/12)*8760):int((9/12)*8760)]   # uncomment to run for a given period only
# #%% res analysis
# # min_res = min(X[:, -1])
# X_new = np.copy(X)
# X_new[:, -1] = X[:, -1] - res_gen_to_remove
# predictions_pre = rf.predict(X_new)
#
# d_p = predictions_pre[:, 0] - predictions[:, 0]
# d_d = predictions_pre[:, 1] - predictions[:, 1]
#
#
# fig = go.Figure(go.Histogram(x=d_p))
# fig.show()
#
#
# print('average increase of price', d_p.mean())
# print('average increase of demand', d_d.mean())
# #todo really bad here
# unique, counts = np.unique(d_p < 0, return_counts=True)
# a = dict(zip(unique, counts))
# print("Number of hours in which the price is not reduced: " + str(a[True]) + ' i.e. % ' + str(100 * a[True] / (a[False] + a[True])))
#
# fig = go.Figure(go.Histogram(x=d_p, nbinsx=20))
# fig.show()
#
# unique, counts = np.unique(d_d > 0, return_counts=True)
# a = dict(zip(unique, counts))
# print("Number of hours in which the quantity has not reduced: " + str(a[True]) + ' i.e. % ' + str(100 * a[True] / (a[False] + a[True])))
#
#
# fig = go.Figure()
# fig.add_trace(go.Scatter(y=predictions_pre[:, 1], mode='lines', name='reduced'))
# fig.add_trace(go.Scatter(y=predictions[:, 1], mode='lines', name='original'))
# fig.update_layout(title_text='Diff in demand:')
# fig.add_trace(go.Scatter(y=res_gen_to_remove, mode='lines', name='res gen to remove'))
# fig.add_trace(go.Scatter(y=X[:, -1] - res_gen_to_remove, mode='lines', name='new res gen'))
# fig.add_trace(go.Scatter(y=X[:, -1], mode='lines', name='org res gen'))
# fig.add_trace(go.Scatter(y=country_data.loc[:, "res"], mode='lines', name='org res gen file'))
# fig.update_xaxes(title_text="hour in year")
# fig.show()
#
# fig = go.Figure()
# fig.add_trace(go.Scatter(y=predictions_pre[:, 0], mode='lines', name='reduced'))
# fig.add_trace(go.Scatter(y=predictions[:, 0], mode='lines', name='original'))
# fig.update_layout(title_text='Diff in Price:')
# fig.update_xaxes(title_text="hour in year")
# fig.show()
#%% bootstrap
from  sklearn.utils import resample

A = 13610  # todo find max A
wind_data = pd.read_csv('French Data/ninja_wind_country_FR_current-merra-2_corrected.csv', index_col=0, header=2)
# ninja_wind_country_FR_current-merra-2_corrected.csv # ninja_wind_country_FR_near-termfuture-merra-2_corrected.csv # ninja_wind_country_FR_long-termfuture-merra-2_corrected.csv
indices = [i for i in wind_data.index if i.startswith(year)]
wind_data_year_tech = wind_data.loc[indices, tech].values
# wind_data_year_tech[:] = 0.5   # do the simulation for a given average availabitliy factor
if np.ndim(wind_data_year_tech) > 1:
    res_gen_to_remove = np.sum(A * wind_data_year_tech, axis=1)
else:
    res_gen_to_remove = A * wind_data_year_tech  # wind_data_year_tech = wind_data_year_tech[int((5/12)*8760):int((9/12)*8760)]   # uncomment to run for a given period only

bootstrap_rounds = 100
X_new = np.copy(X)
X_new[:, -1] = X[:, -1] - res_gen_to_remove
d_p_df = pd.DataFrame(data=np.empty((bootstrap_rounds, 8760)), columns=[h for h in range(8760)],
                      index=[i for i in range(bootstrap_rounds)])
d_q_df = pd.DataFrame(data=np.empty((bootstrap_rounds, 8760)), columns=[h for h in range(8760)],
                      index=[i for i in range(bootstrap_rounds)])

for run in range(bootstrap_rounds):
    print(run)
    X_bt, Y_bt = resample(X, Y, n_samples=len(X), replace=True)
    rf = RandomForestRegressor(n_estimators=10, bootstrap=True, random_state=42, oob_score=False, criterion='mae', n_jobs=10)
    rf.fit(X_bt, Y_bt)
    predictions = rf.predict(X)
    predictions_pre = rf.predict(X_new)
    d_p_df.loc[run, :] = predictions_pre[:, 0] - predictions[:, 0]
    d_q_df.loc[run, :] = predictions_pre[:, 1] - predictions[:, 1]


fig = go.Figure()
fig.add_trace(go.Box(y=d_p_df.mean(axis=1)))
fig.add_trace(go.Box(y=d_q_df.mean(axis=1)))
fig.show()

fig = go.Figure(go.Histogram(x=d_p_df.mean(axis=1), nbinsx=20))
fig.show()
sorted = d_p_df.mean(axis=1).sort_values()
d_p_df.mean(axis=1).mean()
sorted.iloc[4]
sorted.iloc[94]


fig = go.Figure(go.Histogram(x=d_q_df.mean(axis=1), nbinsx=20))
fig.show()

# from sklearn.tree import export_graphviz
# import pydot
# tree = rf.estimators_[1]
# export_graphviz(tree, out_file='tree2.dot', feature_names=columns_to_consider, rounded=True, precision=1)
# graph = pydot.graph_from_dot_file('tree2.dot')
# graph.write_png('tree.png')
#
#
# graph = pydot.graph_from_dot_file('tree2.dot')
# graph.write('tree.png')
#
# importances = list(rf.feature_importances_)
# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(columns_to_consider, importances)]
# feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
# [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
# X, Y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)

# scaling X and Y
# scaler_Y = StandardScaler()
# scaler_Y.fit(Y)
# Y_scaled = scaler_Y.transform(Y)
# scaler_X = StandardScaler()
# scaler_X.fit(X)
# X_scaled = scaler_X.transform(X)
# FR_RD_scaler = 2
# X_scaled[:, 0] = FR_RD_scaler * X_scaled[:, 0]
# # finding KNN design
# neigh = KNeighborsRegressor()
# param_grid = [{'n_neighbors': [5, 15, 52, 168], 'weights': ['uniform']}]
# clf = GridSearchCV(neigh, param_grid, scoring='r2', cv=10, refit=True)  # scoring='neg_mean_squared_error'
# clf.fit(X_scaled, Y_scaled)
#
# # loading all data points (not just the inliers) -------------------------------
# X = country_data.loc[:, columns_to_consider].values.reshape(-1, len(
#     columns_to_consider))  # comment to ignore outliers in the predictions
# X_scaled = scaler_X.transform(X)
# X_scaled[:, 0] = FR_RD_scaler * X_scaled[:, 0]
# Y = country_data.loc[:, ['Price', 'Demand']].values.reshape(-1, 2)  # comment to ignore outliers in the predictions
# Y_scaled = scaler_Y.transform(Y)
# # wind_data_year_tech = wind_data_year_tech[inliers == 1]          #uncomment to ignore outliers in the predictions
#
# # prediction and error calculation for base model
# Y_pred = clf.predict(X_scaled)
# Y_pred_org_scale = scaler_Y.inverse_transform(Y_pred)
# mse = mean_squared_error(Y[:, 0], Y_pred_org_scale[:, 0])
# r2 = r2_score(Y[:, 0], Y_pred_org_scale[:, 0])
# orders = np.argsort(X_scaled[:, 0].flatten())
# X_scaled_sorted = X_scaled[orders]
# X_sorted_org_scale = X[orders]
# Y_pred_ordered = clf.predict(X_scaled_sorted)
# Y_pred_ordered_org_scale = scaler_Y.inverse_transform(Y_pred_ordered)
#
# # prediction and error calculation for reduced residual demand
# X_reduced = np.copy(X)
# X_reduced[:, 0] = X_reduced[:, 0] - res_gen
# X_reduced_scaled = scaler_X.transform(X_reduced)
# X_reduced_scaled[:, 0] = FR_RD_scaler * X_reduced_scaled[:, 0]
# Y_pred_reduced = clf.predict(X_reduced_scaled)
# Y_pred_reduced_org_scale = scaler_Y.inverse_transform(Y_pred_reduced)
# Y_pred_reduced_org_scale = Y_pred_reduced_org_scale[orders]
#
# # calculating differences in Y
# price_dif = Y_pred_ordered_org_scale[:, 0].flatten() - Y_pred_reduced_org_scale[:, 0].flatten()
# price_dif_pos = price_dif[price_dif >= 0]
# quantity_dif = Y_pred_ordered_org_scale[:, 1].flatten() - Y_pred_reduced_org_scale[:, 1].flatten()
# quantity_dif_pos = quantity_dif[quantity_dif >= 0]
# q_final_ordered = Y_pred_ordered_org_scale[:, 1].flatten() - quantity_dif + res_gen[orders]
# price_weighted_FIP = np.average(Y_pred_ordered_org_scale[:, 0].flatten(),
#                                 weights=Y_pred_ordered_org_scale[:, 1].flatten())
# price_weighted_FIT = np.average(Y_pred_reduced_org_scale[:, 0].flatten(), weights=q_final_ordered)
# price_dif_weighted_mean = price_weighted_FIP - price_weighted_FIT
# price_dif_mean = price_dif.mean()
# price_dif_pos_mean = price_dif[price_dif >= 0].mean()
#
# fig = go.Figure(
#     data=[go.Histogram(x=Y_pred_ordered_org_scale[:, 0].flatten() - Y_pred_reduced_org_scale[:, 0].flatten())])
# fig.update_layout(title="(Ignoring negatives) Average weighted price reduction is  " + str(
#     price_dif_weighted_mean) + " (not weighted is " + str(price_dif_pos_mean) + ") vs. original wa of  " + str(
#     price_weighted_FIP))
# fig.write_html("figures/" + country + "_histogram.html")
# fig.show()
#
# mean_reduction_in_quantity = quantity_dif.mean()
# mean_reduction_in_quantity_only_positive = quantity_dif[quantity_dif >= 0].mean()
# fig = go.Figure(
#     data=[go.Histogram(x=Y_pred_ordered_org_scale[:, 1].flatten() - Y_pred_reduced_org_scale[:, 1].flatten())])
# fig.update_layout(title="Average quantity reduction is  " + str(int(mean_reduction_in_quantity)) +
#                         " (" + str(int(mean_reduction_in_quantity_only_positive)) + " ignoring negatives)" +
#                         " while average additional RES generation is " + str(int(res_gen.mean())) +
#                         " making total demand increase of " + str(int(-mean_reduction_in_quantity + res_gen.mean())) +
#                         " (average original Demand is " + str(Y[:, 1].mean()) + ")"
#                   )
# fig.show()
#
# # other figures
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=X[:, 0].flatten(), y=Y[:, 0].flatten(), mode='markers', name='Original'))
# fig.add_trace(go.Scatter(x=X[inliers == 1, 0].flatten(), y=Y[inliers == 1, 0].flatten(), mode='markers',
#                          name='Inliers'))  # comment to ignore outliers in the predictions
# fig.add_trace(go.Scatter(x=X_sorted_org_scale[:, 0].flatten(), y=Y_pred_ordered_org_scale[:, 0].flatten(), mode='lines',
#                          name='Fit:' + str(clf.best_params_['n_neighbors']) + ' R2: ' + str(clf.best_score_)))
# fig.add_trace(go.Scatter(x=X_sorted_org_scale[:, 0].flatten(), y=10 * np.sign(
#     Y_pred_ordered_org_scale[:, 0].flatten() - Y_pred_reduced_org_scale[:, 0].flatten()), mode='lines', name='Error'))
# fig.add_trace(go.Scatter(x=X_sorted_org_scale[:, 0].flatten(), y=res_gen[orders] / 1000, mode='lines', name='New RD'))
# fig.add_trace(go.Scatter(x=X_sorted_org_scale[:, 0].flatten(), y=Y_pred_reduced_org_scale[:, 0].flatten(), mode='lines',
#                          name='Fit:' + str(clf.best_params_['n_neighbors']) + ' Reduced'))
# unique, counts = np.unique((Y_pred_ordered_org_scale[:, 0] - Y_pred_reduced_org_scale[:, 0]) > 0, return_counts=True)
# a = dict(zip(unique, counts))
# fig.update_layout(
#     title="Number of hours in which the price is not reduced: " + str(a[False]) + ' i.e. % ' + str(
#         100 * a[False] / (a[False] + a[True]))
# )
# fig.show()
# fig.write_html("figures/" + country + "_KKN_1000.html")
#
# fig = ff.create_distplot([Y[:, 0][orders] - Y_pred_ordered_org_scale[:, 0].flatten()], ["group_labels"],
#                          curve_type='normal')
# fig.show()
#
# # fig = go.Figure(go.Histogram2d(x=X_sorted_org_scale[:, 0].flatten(), y=Y_pred_ordered_org_scale[:, 0].flatten(
# # )-Y_pred_reduced_org_scale[:, 0].flatten()))
# fig = go.Figure(go.Scatter(x=X_sorted_org_scale[:, 0].flatten(),
#                            y=Y_pred_ordered_org_scale[:, 0].flatten() - Y_pred_reduced_org_scale[:, 0].flatten(),
#                            mode='markers', name='Original'))
# fig.update_layout(title="Residual Demand vs. DPrice")
# fig.show()
#
# # fig = go.Figure(go.Histogram2d(x=X_sorted_org_scale[:, 0].flatten(),
# # y=Y_pred_ordered_org_scale[:, 0].flatten()-Y_pred_reduced_org_scale[:, 0].flatten()))
# # seasons_No = 6
# # fig = go.Figure()
# # for i in range(seasons_No):
# #     start_hour = int((i/seasons_No)*8760)
# #     end_hour = int(((i+1)/seasons_No)*8760)
# #     fig.add_trace(go.Scatter(x=X[start_hour:end_hour, 0].flatten(),
# #     y=Y_pred_org_scale[start_hour:end_hour, 0].flatten()-
# #     scaler_Y.inverse_transform(Y_pred_reduced)[start_hour:end_hour,0], mode='markers', name=str(i)))
# # fig.update_layout(title="Residual Demand vs. DPrice in seasons")
# # # fig.update_yaxes(range=[-2, 0])
# # # fig.update_xaxes(range=[20000, 80000])
# # fig.show()
#
# fig = go.Figure(go.Histogram2d(x=res_gen[orders],
#                                y=Y_pred_ordered_org_scale[:, 0].flatten() - Y_pred_reduced_org_scale[:, 0].flatten()))
# # fig = go.Figure(go.Scatter(x=wind_data_year_tech[orders],
# # y=Y_pred_ordered_org_scale[:, 0].flatten()-Y_pred_reduced_org_scale[:, 0].flatten(), mode='markers', name='Original'))
# fig.update_layout(title="RES vs. DPrice")
# fig.show()
#
# xx = X_sorted_org_scale.sum(axis=1)
# fig = go.Figure(
#     go.Histogram2d(x=xx, y=Y_pred_ordered_org_scale[:, 0].flatten() - Y_pred_reduced_org_scale[:, 0].flatten()))
# # fig = go.Figure(go.Scatter(x=xx, y=Y_pred_ordered_org_scale[:, 0].flatten()-Y_pred_reduced_org_scale[:, 0].flatten(),
# # mode='markers', name='Original'))
# fig.update_layout(title="Sum neighbours vs. DPrice")
# fig.show()
#
# print(clf.cv_results_)
# print(clf.best_estimator_)
# print(clf.best_score_)
# print(clf.best_params_)
# print(clf.scorer_)
