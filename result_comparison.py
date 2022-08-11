#%%
from matplotlib.pyplot import axes
import plotly.io as pio
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import pickle
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

pio.renderers.default = 'browser'
#%%
def w_avg(df_p, df_q):
    ind_name_list = df_p.index
    p_weighted = pd.DataFrame(index=ind_name_list,columns=["w_avg"])
    for ind_name in ind_name_list:
        p_weighted.loc[ind_name,"w_avg"] = sum(df_p.loc[ind_name,:]*df_q.loc[ind_name,:])/sum(df_q.loc[ind_name,:])
    return p_weighted
#%% Import data from pickle
results_dic = {}
scenario_list = ["AF_nuts_FIP_newcomer_Allow_mult_cap5_scaled","AF_nuts_FIP_newcomer_Allow_mult_cap9_scaled", "AF_nuts_FIP_newcomer_Allow_mult_cap9",  "AF_nuts_FIP_incumbent"]  # "FR_avg",  "FR_nuts", "FR_FIP_Allow_mult_cap3", "FR_FIP_Allow_mult_cap6",  "FR_FIP_Allow_mult_cap9_old_probably_incumbent"
for scenario in scenario_list + ["AF_nuts_FIP_newcomer_Allow_mult_cap1"]:
    print(scenario)
    with open('variables_'+ scenario +'.pkl', 'rb') as f:
        d_p_predic_df, d_q_predic_df, res_gen_to_remove, country_data, X, Y, X_new, rf,\
        wind_data_year_tech, res_gen_to_remove,\
        p_predict_df, q_predict_df, p_red_predict_df, q_red_predict_df, wind_cap_FIP_nuts2 = pickle.load(f)
        results_dic[scenario]={"d_p_predic_df": d_p_predic_df, "d_q_predic_df": d_q_predic_df, "res_gen_to_remove": res_gen_to_remove, "country_data": country_data, "X": X, "Y": Y, "X_new": X_new, "rf": rf, "wind_data_year_tech": wind_data_year_tech, "res_gen_to_remove": res_gen_to_remove, "p_predict_df": p_predict_df, "q_predict_df": q_predict_df, "p_red_predict_df": p_red_predict_df, "q_red_predict_df": q_red_predict_df, "wind_cap_FIP_nuts2": wind_cap_FIP_nuts2}

#%% Figures - RES output
fig = go.Figure()
for scenario in scenario_list:
    fig.add_trace(go.Scatter(y=results_dic[scenario]["res_gen_to_remove"], mode='lines', name=scenario + ' - Average [{:.1f} TWh]'.format(results_dic[scenario]["res_gen_to_remove"].sum()/1000000)))
fig.update_layout(title_text='RES generation (to be removed from the system)')
fig.show()
#%% Figure - Price
d_p_df_mean = d_p_predic_df.mean(axis=1).mean()

p_av = np.mean(Y[:,0])

fig = go.Figure()
fig.add_hline(y=p_av, annotation_text='Avg: {:.2f}' .format(p_av))
for scenario in scenario_list:
    values_to_plot = results_dic[scenario]["p_red_predict_df"].mean(axis=1)
    fig.add_trace(go.Box(y=values_to_plot, name="Price "+ scenario  + ' {:.2f}' .format(values_to_plot.values.mean())
                                                        +    ' ({:.2f} %)' .format(100*(values_to_plot.values.mean()-p_av)/p_av)
                                                        + ' CI=[{:.2f}' .format(np.quantile(values_to_plot.values,0.025)) 
                                                        +   ', {:.2f}' .format(np.quantile(values_to_plot.values,0.975)) + ']'
    
                        )
                  )
fig.show()
#%% price error bars
p_av_actual = sum(Y[:,0]*Y[:,1])/sum(Y[:,1])

mean_val_list = []
perc_02_5_list = []
perc_97_5_list = []
fig = go.Figure()
for scenario in scenario_list:
    # values_to_plot = results_dic[scenario]["p_red_predict_df"].mean(axis=1)
    values_to_plot = w_avg(results_dic[scenario]["p_red_predict_df"], results_dic[scenario]["q_red_predict_df"]).w_avg
    mean_val = values_to_plot.mean(axis=0)
    mean_val_list.append(mean_val)
    perc_02_5 = np.quantile(values_to_plot.values,0.025)
    perc_02_5_list.append(perc_02_5)
    perc_97_5 = np.quantile(values_to_plot.values,0.975) 
    perc_97_5_list.append(perc_97_5)
# mode='markers'
fig.add_trace(go.Scatter(
    y=np.array(mean_val_list),
    mode='markers',
    error_y=dict(
        type='data',
        symmetric=False,
        array=np.array(mean_val_list)-np.array(perc_02_5_list),
        arrayminus=np.array(perc_97_5_list)-np.array(mean_val_list)
    )))
fig.add_hline(y=p_av_actual, annotation_text='Avg: {:.2f}' .format(p_av_actual))
fig.show()
#%% Figure - Price  p_predict_df
p_av_actual = sum(Y[:,0]*Y[:,1])/sum(Y[:,1])
fig = go.Figure()
for scenario in scenario_list:
    weighted_price = w_avg(results_dic[scenario]["p_red_predict_df"], results_dic[scenario]["q_red_predict_df"])
    fig.add_trace(go.Box(y=weighted_price.mean(axis=1), name="Price "+ scenario + ' {:.2f}' .format(weighted_price.values.mean()) 
                                                        +  ' ({:.2f} %)' .format(100*(weighted_price.values.mean()-p_av_actual)/p_av_actual)
                                                        + ' CI=[{:.2f}' .format(np.quantile(weighted_price.values,0.025)) 
                                                        +   ', {:.2f}' .format(np.quantile(weighted_price.values,0.975)) + ']'))
fig.add_hline(y=p_av_actual, annotation_text='Avg: {:.2f}' .format(p_av_actual))
fig.update_yaxes(range=[40, 44])
fig.show()
#%% Figure - Consumption
fig = go.Figure()
for scenario in scenario_list:
    fig.add_trace(go.Box(y=results_dic[scenario]["d_q_predic_df"].mean(axis=1), name="Price "+ scenario))
    # fig.update_layout(title_text='Range of difference estimates in bootstrap method for price increase: {:.2f}'.format(results_dic[scenario]["d_p_df_mean"])
    #                             + ' ({:.2f}'.format(100*results_dic[scenario]["d_p_df_mean"]/country_data.loc[:, "Price"].values.mean()) + '%) CI=[{:.2f}'. format(results_dic[scenario]["d_q_predic_df"].mean(axis=1).quantile(q=0.025))
    #                             + ', {:.2f}'.format(results_dic[scenario]["d_q_predic_df"].mean(axis=1).quantile(q=0.975)) + ']')
fig.show()
#%% Figure - price weighted average
d_p_df_mean = d_p_predic_df.mean(axis=1).mean()
fig = go.Figure()
for scenario in scenario_list:
    fig.add_trace(go.Box(y=results_dic[scenario]["d_p_predic_df"].mean(axis=1), name="Price "+ scenario))
fig.show()

# %%
# average_price = predictions[:, 0] * predictions[:, 1]

# #%%
# rf.fit(X_bt, Y_bt)
# predictions = rf.predict(X)
# predictions_pre = rf.predict(X_new)
# d_p_predic_df.loc[run, :] = predictions_pre[:, 0] - predictions[:, 0]
# d_q_predic_df.loc[run, :] = predictions_pre[:, 1] - predictions[:, 1]
# p_predict_df.loc[run, :] = predictions[:, 0]
# q_predict_df.loc[run, :] = predictions[:, 1]
# p_red_predict_df.loc[run, :] = predictions_pre[:, 0]
# q_red_predict_df.loc[run, :] = predictions_pre[:, 1]