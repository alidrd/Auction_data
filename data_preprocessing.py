import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import plotly.express as px
from hourly_maker import hourly_maker_fcn


#%% data preparation
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
months_order = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
bidding_zone = {'DE': 'DE_LU', 'CH': 'CH', 'FR': 'FR', 'GB': 'GB', 'ES': 'ES', 'BE': 'BE', 'IT': 'IT_North'}
hours_in_time_res = {'PT15M': 4, 'PT30M': 2, 'PT60M': 1}

base_country = 'GB'

# df_gen_sample = pd.read_csv("Data/parsed_data/jan2019AggregatedGenerationPerType.csv")
# df_gen_sample = df_gen_sample[(df_gen_sample.AreaTypeCode == "CTY") & (df_gen_sample.MapCode == base_country) & (df_gen_sample.MapCode == base_country) & (df_gen_sample.ProductionType == 'Nuclear')]
# df_gen_sample = df_gen_sample.sort_values(by="DateTime")
# df_gen_sample = hourly_maker_fcn(df_gen_sample, 'ActualGenerationOutput')


df_M_X_sample = pd.read_csv("Data/parsed_data/jan2019CrossBorderPhysicalFlow.csv")
neighbours = df_M_X_sample[(df_M_X_sample.OutAreaTypeCode == "CTY") & (df_M_X_sample.InMapCode == base_country)].OutMapCode.unique()
if neighbours[0] == 'GB':
    neighbours = np.delete(neighbours, 0)
    neighbours = np.append(neighbours, ['GB'])
print('Neighbours are ', neighbours)
M_columns = ['Imported_' + country for country in neighbours]
X_columns = ['Exported_' + country for country in neighbours]
df_gen = pd.read_csv("Data/parsed_data/jan2019AggregatedGenerationPerType.csv")
technologies = df_gen.ProductionType.unique()
columns = ['Price', 'Demand', 'Demand_pump_storage']
columns.extend(M_columns)
columns.extend(X_columns)
columns.extend(technologies)
data_FR = pd.DataFrame(columns=columns)
start = 1
#%% starting monthly loop
for month in months:
    print(month, 80*'-')
    temp_month = pd.DataFrame()

#%% import export
    M_X_raw = pd.read_csv("Data/parsed_data/" + month + "2019CrossBorderPhysicalFlow.csv")
    M_X_raw = M_X_raw.sort_values(by="DateTime")
    for country in neighbours:
        df_X = M_X_raw[(M_X_raw.OutAreaTypeCode == "CTY") & (M_X_raw.OutMapCode == base_country) & (M_X_raw.InMapCode == country)]
        df_X = hourly_maker_fcn(df_X, 'FlowValue')
        try:
            temp_month["Exported_" + country] = df_X.FlowValue.values
        except ValueError:
            print('Error in the size of data in Exported_' + country, 'Starting DateTime approach')
            temp_month["Exported_" + country] = np.nan
            temp_month_country = df_X
            no_of_rows_for_country = len(temp_month_country)
            col_cntr = 0
            hour_in_month = 0
            while col_cntr < no_of_rows_for_country:
                day = hour_in_month // 24 + 1
                hour_in_day = hour_in_month % 24
                if temp_month_country.iloc[col_cntr, :].DateTime == str(datetime.datetime(2019, months_order[month], day, hour_in_day)) + '.000': #date not missing
                    temp_month.loc[:, "Exported_" + country].iloc[hour_in_month] = temp_month_country.iloc[col_cntr, :].FlowValue
                    col_cntr = col_cntr + 1
                    hour_in_month = hour_in_month + 1
                else:
                    hour_in_month = hour_in_month + 1
        df_M = M_X_raw[(M_X_raw.OutAreaTypeCode == "CTY") & (M_X_raw.OutMapCode == country) & (M_X_raw.InMapCode == base_country)]
        df_M = hourly_maker_fcn(df_M, 'FlowValue')
        try:
            temp_month["Imported_" + country] = df_M.FlowValue.values
        except ValueError:
            print('Error in the size of data in Imported_' + country, 'Starting DateTime approach')
            temp_month["Imported_" + country] = np.nan
            temp_month_country = df_M
            no_of_rows_for_country = len(temp_month_country)
            col_cntr = 0
            hour_in_month = 0
            while col_cntr < no_of_rows_for_country:
                day = hour_in_month // 24 + 1
                hour_in_day = hour_in_month % 24
                if temp_month_country.iloc[col_cntr, :].DateTime == str(datetime.datetime(2019, months_order[month], day, hour_in_day)) + '.000': #date not missing
                    temp_month.loc[:, "Imported_" + country].iloc[hour_in_month] = temp_month_country.iloc[col_cntr, :].FlowValue
                    col_cntr = col_cntr + 1
                    hour_in_month = hour_in_month + 1
                else:
                    hour_in_month = hour_in_month + 1

#%% prices
    df_P = pd.read_csv("Data/parsed_data/" + month + "2019DayAheadPrices.csv")
    df_P = df_P.sort_values(by="DateTime")
    temp_month['Price'] = df_P[df_P.MapCode == bidding_zone[base_country]].Price.values
#%% load
    #df_D = pd.read_csv("Data/parsed_data/" + month + "2019DayAheadTotalLoadForecast.csv")
    D_raw = pd.read_csv("Data/parsed_data/" + month + "2019ActualTotalLoad.csv")
    df_D = D_raw[(D_raw.AreaTypeCode == "CTY") & (D_raw.MapCode == base_country)]
    df_D = df_D.sort_values(by="DateTime")
    df_D = hourly_maker_fcn(df_D, 'TotalLoadValue')
    try:
        temp_month['Demand'] = df_D.TotalLoadValue.values
    except ValueError:
        print("Error in loading demand for ", month)
        temp_month['Demand'] = np.nan
        temp_month_country = df_D
        no_of_rows_for_country = len(temp_month_country)
        col_cntr = 0
        hour_in_month = 0
        while col_cntr < no_of_rows_for_country:
            day = hour_in_month // 24 + 1
            hour_in_day = hour_in_month % 24
            if temp_month_country.iloc[col_cntr, :].DateTime == str(
                    datetime.datetime(2019, months_order[month], day, hour_in_day)) + '.000':  # date not missing
                temp_month.loc[:, 'Demand'].iloc[hour_in_month] = temp_month_country.iloc[col_cntr, :].TotalLoadValue
                col_cntr = col_cntr + 1
                hour_in_month = hour_in_month + 1
            else:
                hour_in_month = hour_in_month + 1

#%% generation
    gen_raw = pd.read_csv("Data/parsed_data/" + month + "2019AggregatedGenerationPerType.csv")
    gen_raw = gen_raw.sort_values(by="DateTime")
    for technology in technologies:
        df_gen = gen_raw[(gen_raw.AreaTypeCode == "CTY") & (gen_raw.MapCode == base_country) & (gen_raw.ProductionType == technology)]
        if len(df_gen) == 0: #no such technology exists
            temp_month[technology] = 0
            print(technology, ' not in ', base_country)
        else:
            df_gen = hourly_maker_fcn(df_gen, 'ActualGenerationOutput')
            try:
                temp_month[technology] = df_gen.ActualGenerationOutput.values
            except ValueError:
                print("Error in the size of Generation for technoloyg ", technology)
                if len(df_gen[df_gen.ProductionType == technology].ActualGenerationOutput.values) == 0:
                    print(technology, " does not exist in ", base_country)
                    temp_month[technology] = 0
                else:
                    temp_month[technology] = np.nan
                    temp_month_country = df_gen
                    no_of_rows_for_country = len(temp_month_country)
                    col_cntr = 0
                    hour_in_month = 0
                    while col_cntr < no_of_rows_for_country:
                        day = hour_in_month // 24 + 1
                        hour_in_day = hour_in_month % 24
                        if temp_month_country.iloc[col_cntr, :].DateTime == str(datetime.datetime(2019, months_order[month], day, hour_in_day)) + '.000':  # date not missing
                            temp_month.loc[:, technology].iloc[hour_in_month] = temp_month_country.iloc[col_cntr, :].ActualGenerationOutput
                            col_cntr = col_cntr + 1
                            hour_in_month = hour_in_month + 1
                        else:
                            hour_in_month = hour_in_month + 1
#%% demand pump storage
    df_gen = gen_raw[(gen_raw.AreaTypeCode == "CTY") & (gen_raw.MapCode == base_country) & (gen_raw.ProductionType == 'Hydro Pumped Storage')]
    df_gen = hourly_maker_fcn(df_gen, 'ActualConsumption')
    try:
        if df_gen.ActualConsumption.isna().all():
            temp_month['Demand_pump_storage'] = 0
        else:
            temp_month['Demand_pump_storage'] = df_gen.ActualConsumption.values
    except ValueError:
        if len(df_gen.ActualConsumption.values) == 0 or base_country == 'CH':
            temp_month['Demand_pump_storage'] = 0
        else:
            print("Error in the size of Demand pump storage")
            temp_month["Demand_pump_storage"] = np.nan
            temp_month_country = df_gen
            no_of_rows_for_country = len(temp_month_country)
            col_cntr = 0
            hour_in_month = 0
            while col_cntr < no_of_rows_for_country:
                day = hour_in_month // 24 + 1
                hour_in_day = hour_in_month % 24
                if temp_month_country.iloc[col_cntr, :].DateTime == str(datetime.datetime(2019, months_order[month], day, hour_in_day)) + '.000':  # date not missing
                    temp_month.loc[:, "Demand_pump_storage"].iloc[hour_in_month] = temp_month_country.iloc[col_cntr,:].ActualConsumption
                    col_cntr = col_cntr + 1
                    hour_in_month = hour_in_month + 1
                else:
                    hour_in_month = hour_in_month + 1
#%% appending months
    data_FR = data_FR.append(temp_month)

data_FR.to_csv('temp4.csv')
#data_FR = data_FR.fillna(0)
data_FR['Balance'] = data_FR.sum(axis=1, skipna=True)
negative_variables = ['Demand', 'Demand_pump_storage']
negative_variables.extend(X_columns)

data_FR.loc[:, 'Balance'] = data_FR.loc[:, 'Balance'] - data_FR.loc[:, 'Price']
for negative_variable in negative_variables:
    data_FR.loc[:, 'Balance'] = data_FR.loc[:, 'Balance'] - 2 * data_FR.loc[:, negative_variable]

data_FR["hour"] = [i for i in range(len(data_FR))]
data_FR.index = [i for i in range(len(data_FR))]
data_FR.to_csv(base_country +'_data.csv')

df1 = data_FR.melt(id_vars=['hour']+list(data_FR.keys()[len(data_FR):]), var_name='Time Series')
fig = px.line(df1, x='hour', y='value', color='Time Series')
fig.show()