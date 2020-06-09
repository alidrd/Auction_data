import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import plotly.express as px


#%% data preparation
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
months_order = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
df_M_X = pd.read_csv("Data/parsed_data/jan2019DayAheadCommercialSchedules.csv")
base_country = 'FR'
neighbours = df_M_X[(df_M_X.OutAreaTypeCode == "CTY") & (df_M_X.InMapCode == base_country)].OutMapCode.unique()
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
    temp_month = pd.DataFrame()

#%% import export
    df_M_X = pd.read_csv("Data/parsed_data/" + month + "2019DayAheadCommercialSchedules.csv")
    df_M_X = df_M_X.sort_values(by="DateTime")
    for country in neighbours:
        try:
            temp_month["Exported_" + country] = df_M_X[(df_M_X.OutAreaTypeCode == "CTY")
                                                        & (df_M_X.OutMapCode == 'FR')
                                                        & (df_M_X.InMapCode == country)].Capacity.values
        except ValueError:
            print('Error in the size of data in Exported_' + country, 'Starting DateTime approach')
            temp_month["Exported_" + country] = np.nan
            temp_month_country = df_M_X[(df_M_X.OutAreaTypeCode == "CTY")
                                        & (df_M_X.OutMapCode == 'FR')
                                        & (df_M_X.InMapCode == country)]
            no_of_rows_for_country = len(temp_month_country)
            col_cntr = 0
            hour_in_month = 0
            while col_cntr < no_of_rows_for_country:
                day = hour_in_month // 24 + 1
                hour_in_day = hour_in_month % 24
                if temp_month_country.iloc[col_cntr, :].DateTime == str(datetime.datetime(2019, months_order[month], day, hour_in_day)) + '.000': #date not missing
                    temp_month.loc[:, "Exported_" + country].iloc[hour_in_month] = temp_month_country.iloc[col_cntr, :].Capacity
                    col_cntr = col_cntr + 1
                    hour_in_month = hour_in_month + 1
                else:
                    hour_in_month = hour_in_month + 1
        try:
            temp_month["Imported_" + country] = df_M_X[(df_M_X.OutAreaTypeCode == "CTY")
                                                       & (df_M_X.OutMapCode == country)
                                                       & (df_M_X.InMapCode == 'FR')].Capacity.values
        except ValueError:
            print('Error in the size of data in Imported_' + country, 'Starting DateTime approach')
            temp_month["Imported_" + country] = np.nan
            temp_month_country = df_M_X[(df_M_X.OutAreaTypeCode == "CTY")
                                        & (df_M_X.OutMapCode == country)
                                        & (df_M_X.InMapCode == 'FR')]
            no_of_rows_for_country = len(temp_month_country)
            col_cntr = 0
            hour_in_month = 0
            while col_cntr < no_of_rows_for_country:
                day = hour_in_month // 24 + 1
                hour_in_day = hour_in_month % 24
                if temp_month_country.iloc[col_cntr, :].DateTime == str(datetime.datetime(2019, months_order[month], day, hour_in_day)) + '.000': #date not missing
                    temp_month.loc[:, "Imported_" + country].iloc[hour_in_month] = temp_month_country.iloc[col_cntr, :].Capacity
                    col_cntr = col_cntr + 1
                    hour_in_month = hour_in_month + 1
                else:
                    hour_in_month = hour_in_month + 1

#%% prices
    df_P = pd.read_csv("Data/parsed_data/" + month + "2019DayAheadPrices.csv")
    df_P = df_P.sort_values(by="DateTime")
    temp_month['Price'] = df_P[df_P.MapCode=="FR"].Price.values
#%% load
    #df_D = pd.read_csv("Data/parsed_data/" + month + "2019DayAheadTotalLoadForecast.csv")
    df_D = pd.read_csv("Data/parsed_data/" + month + "2019ActualTotalLoad.csv")
    df_D = df_D[(df_D.AreaTypeCode == "CTY") & (df_D.MapCode == "FR")]
    df_D = df_D.sort_values(by="DateTime")
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
    df_gen = pd.read_csv("Data/parsed_data/" + month + "2019AggregatedGenerationPerType.csv")
    df_gen = df_gen[(df_gen.AreaTypeCode == "CTY") & (df_gen.MapCode == "FR")]
    df_gen = df_gen.sort_values(by="DateTime")
    for technology in technologies:
        try:
            temp_month[technology] = df_gen[df_gen.ProductionType == technology].ActualGenerationOutput.values
        except ValueError:
            temp_month[technology] = np.nan
            temp_month_country = df_gen[df_gen.ProductionType == technology]
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
    try:
        temp_month['Demand_pump_storage'] = df_gen[df_gen.ProductionType == 'Hydro Pumped Storage'].ActualConsumption.values
    except ValueError:
        temp_month["Demand_pump_storage"] = np.nan
        temp_month_country = df_gen[df_gen.ProductionType == 'Hydro Pumped Storage']
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
data_FR = data_FR.fillna(0)
data_FR['Balance'] = data_FR.sum(axis=1, skipna=True)
negative_variables = ['Demand', 'Demand_pump_storage']
negative_variables.extend(X_columns)

data_FR.loc[:, 'Balance'] = data_FR.loc[:, 'Balance'] - data_FR.loc[:, 'Price']
for negative_variable in negative_variables:
    data_FR.loc[:, 'Balance'] = data_FR.loc[:, 'Balance'] - 2 * data_FR.loc[:, negative_variable]

data_FR["hour"] = [i for i in range(8760)]
data_FR.index = [i for i in range(8760)]
data_FR.to_csv(base_country +'_data.csv')

df1 = data_FR.melt(id_vars=['hour']+list(data_FR.keys()[len(data_FR):]), var_name='Time Series')
fig = px.line(df1, x='hour', y='value', color='Time Series' )
fig.show()