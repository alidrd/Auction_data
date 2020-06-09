import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


neighbours = ['DE', 'BE', 'CH', 'ES', 'GB', "IT"]
M_columns = ['Imported_' + country for country in neighbours]
X_columns = ['Exported_' + country for country in neighbours]

data_FR = pd.read_csv("temp10.csv", index_col=0)
must_runs = ['Solar', 'Biomass', 'Hydro Run-of-river and poundage', 'Waste', 'Wind Offshore']
must_runs.extend(M_columns)

data_FR['Residual_Demand'] = data_FR.loc[:, 'Demand']
data_FR['sum_negative_var'] = 0
for negative_variable in must_runs:
    data_FR.loc[:, 'Residual_Demand'] = data_FR.loc[:, 'Residual_Demand'] - data_FR.loc[:, negative_variable]
    data_FR['sum_negative_var'] = data_FR['sum_negative_var'] - data_FR.loc[:, negative_variable]

x = np.arange(start=data_FR.sum_negative_var.min(), stop=data_FR.sum_negative_var.max(), step=100)
fig = go.Figure()
fig.add_trace(go.Scatter(x=data_FR.sum_negative_var, y=data_FR.Balance, mode='markers', name='Demand'))
fig.add_trace(go.Scatter(x=x, y=x, mode='lines', name='Demand'))
fig.show()



extra_demands = []
extra_demands.extend(X_columns)
# for positive_variable in extra_demands:
#     data_FR.loc[:, 'Residual_Demand'] = data_FR.loc[:, 'Residual_Demand'] + data_FR.loc[:, positive_variable]

#%% Linear Regression
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(data_FR.Residual_Demand.values.reshape(-1, 1), data_FR.Price.values.reshape(-1, 1))  # perform linear regression
Y_pred_Residual_Demand = linear_regressor.predict(data_FR.Residual_Demand.values.reshape(-1, 1))  # make predictions

linear_regressor.fit(data_FR.Demand.values.reshape(-1, 1), data_FR.Price.values.reshape(-1, 1))  # perform linear regression
Y_pred_Demand = linear_regressor.predict(data_FR.Demand.values.reshape(-1, 1))  # make predictions

fig = go.Figure()
fig.add_trace(go.Scatter(x=data_FR.Residual_Demand, y=data_FR.Price, mode='markers', name='Residual'))
fig.add_trace(go.Scatter(x=data_FR.Demand, y=data_FR.Price, mode='markers', name='Demand'))
fig.add_trace(go.Scatter(x=data_FR.Residual_Demand, y=Y_pred_Residual_Demand[:, 0], mode='lines', name='Residual demand'))
fig.add_trace(go.Scatter(x=data_FR.Demand, y=Y_pred_Demand[:, 0], mode='lines', name='Demand'))
fig.show()

data_FR["Season"] = "Spring"
length = len(data_FR)
parts = 4
for part in range(parts):
    data_FR.iloc[int((part / parts) * length):int(((part + 1) / parts) * length), data_FR.columns.get_loc('Season')] = part

#%% box plot
# fig = go.Figure()
# part_devided_to = 10
# for part in range(parts):
#     X = data_FR[data_FR.Season == part].Residual_Demand
#     Y = data_FR[data_FR.Season == part].Price
#     length_X = X.len()
#     X_sorted = X.sort()
#     X_
#
#
#     fig.add_trace(go.Box(y=y0))
#     fig.add_trace(go.Box(y=y1))
#
# fig.show()

#%% K nearest neighbours
fig = go.Figure()
d_reduced = 10000
for part in range(parts):
    X = data_FR[data_FR.Season == part].Residual_Demand.values.reshape(-1, 1)
    Y = data_FR[data_FR.Season == part].Price.values.reshape(-1, 1)
    neigh = KNeighborsRegressor(n_neighbors=50, weights='uniform')
    neigh.fit(X, Y)
    Y_pred_Residual_Demand = neigh.predict(X)
    Y_pred = neigh.predict(np.sort(data_FR.Residual_Demand.values).reshape(-1, 1))
    Y_pred_reduced = neigh.predict(np.sort(data_FR.Residual_Demand.values-d_reduced).reshape(-1, 1))

    fig.add_trace(go.Scatter(x=X.flatten(), y=Y.flatten(), mode='markers', name='Original'))
    fig.add_trace(go.Scatter(x=np.sort(data_FR.Residual_Demand.values).reshape(-1, 1).flatten(), y=Y_pred.flatten(), mode='lines',name='Fit'))
    fig.add_trace(go.Scatter(x=np.sort(data_FR.Residual_Demand.values-d_reduced).reshape(-1, 1).flatten(), y=Y_pred_reduced.flatten(), mode='lines', name='Fit reduced'))
fig.show()

#%% Part plotting
seasons_dic = {"Summer": 24*(31+28+31+30+31), "Fall": 24*(31+28+31+30+31+30+30+31), "Winter": 24*(31+28+31+30+31+30+30+31+30+31+30)} #starting date of seasons in year
data_FR["Season"] = "Spring"
length = len(data_FR)
parts = 12
for part in range(parts):
    data_FR.iloc[int((part / parts) * length):int(((part + 1) / parts) * length - 1), data_FR.columns.get_loc('Season')] = part

fig = go.Figure()
for part in range(parts):
    x = data_FR[data_FR.Season == part].Residual_Demand
    y = data_FR[data_FR.Season == part].Price
    fig.add_trace(go.Scatter(x=x, y=y,
                        mode='markers',
                        name=str(part)))
fig.update_xaxes(range=[data_FR.Residual_Demand.min(), data_FR.Residual_Demand.max()])
fig.show()

