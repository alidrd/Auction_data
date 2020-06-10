def hourly_maker_fcn(df, col_name):
    columns_names = df.columns
    if df.ResolutionCode.iloc[0] == 'PT60M': #all
        final_df = df
    else:
        new_df = df.copy()
        for ind in df.index:
            new_df.loc[ind, 'hour_DateTime'] = new_df.loc[ind, 'DateTime'][0:14] + '00:00.000'
        new_df = new_df.drop(columns= 'DateTime')
        new_df = new_df.rename(columns={"hour_DateTime": 'DateTime'})
        new_df = new_df.reindex(columns=columns_names)
        unique_times = new_df.DateTime.unique()
        final_df = new_df.drop_duplicates(subset='DateTime').copy()
        final_df['ResolutionCode'] = 'PT60M'
        for time in unique_times:
            final_df.loc[final_df.DateTime == time, col_name] = new_df.loc[new_df.DateTime == time, col_name].mean()
    return final_df