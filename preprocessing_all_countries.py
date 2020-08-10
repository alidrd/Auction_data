import pandas as pd
import plotly.express as px

# missing data handling
nan_handling = 'interpolate'
columns_to_process = ['Demand', 'Nuclear', 'Fossil Gas', 'Wind Onshore', 'Wind Offshore',
                      'Hydro Run-of-river and poundage', 'Biomass', 'Fossil Brown coal/Lignite',
                      'Fossil Hard coal', 'Geothermal', ]
#
bidding_zone = {'DE': 'DE_LU', 'CH': 'CH', 'FR': 'FR', 'GB': 'GB', 'ES': 'ES', 'BE': 'BE', 'IT': 'IT_North'}
countries = bidding_zone.keys()

must_runs = ['Wind Onshore', 'Solar', 'Waste', 'Hydro Run-of-river and poundage', 'Wind Offshore', 'Marine', 'Other renewable']  #todo
# 'Hydro Water Reservoir'  'Hydro Pumped Storage' 'Nuclear' 'Biomass'


for country in bidding_zone:
    # missing values    ------------------------------------------------------------------------------------------------
    country_data = pd.read_csv(country + '_data_before_balance.csv', header=0, index_col=0)
    if nan_handling == 'interpolate':
        for column in columns_to_process:
            if column in country_data.columns:
                country_data.loc[:, column] = country_data.loc[:, column].interpolate(method='linear')
    country_data.loc[:, country_data.columns != 'Demand'] = country_data.loc[:, country_data.columns != 'Demand'].fillna(value=0)
    # calculate balance (might be used) --------------------------------------------------------------------------------
    export_countries = [i for i in country_data.columns if i.startswith('Exported_')]
    negative_variables = ['Demand', 'Demand_pump_storage']
    negative_variables.extend(export_countries)
    positive_variables = [element for element in country_data.columns if element not in negative_variables]
    positive_variables.remove('Price')
    country_data['Balance'] = country_data.loc[:, positive_variables].sum(axis=1, skipna=True, min_count=len(positive_variables)) - \
                              country_data.loc[:, negative_variables].sum(axis=1, skipna=True, min_count=len(negative_variables))
    # residual demand ------------------------------------------------------------------------------------------------
    import_countries = [i for i in country_data.columns if i.startswith('Imported_')]
    # must_runs.extend(import_countries)
    country_data['Residual_Demand'] = country_data.loc[:, 'Demand']
    for neg_var in must_runs:
        if neg_var in country_data.columns:
            country_data.loc[:, 'Residual_Demand'] = country_data.loc[:, 'Residual_Demand'] - country_data.loc[:, neg_var]
    # import_countries = [i for i in country_data.columns if i.startswith('Imported_')]


    country_data["hour"] = [i + 1 for i in range(len(country_data))]
    country_data.index = [i + 1 for i in range(len(country_data))]
    country_data.to_csv(country + '_data_missing_data_handeled.csv')

    df1 = country_data.melt(id_vars=['hour'], var_name='Time Series')
    fig = px.line(df1, x='hour', y='value', color='Time Series')
    fig.update_layout(title_text='Country: ' + country + ' with MAE balance of ' + str(country_data.Balance.abs().mean()))
    # fig.show()
    fig.write_html("figures/" + country + "_all_time_series_missing_data_handeled.html")

