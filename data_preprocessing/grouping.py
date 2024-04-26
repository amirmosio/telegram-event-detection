import math
def merge_consecutive_messages(df,timeframe):
    new_df = df.copy()
    k = new_df.shape[0]-1
    i=0
    while i < k-1:
        j=i+1
        delta_timestamp = (new_df.iloc[j]['date']-new_df.iloc[i]['date']).seconds
        if(not math.isnan(new_df.iloc[i]['sender'] )):
            while(new_df.iloc[i]['sender'] == new_df.iloc[j]['sender'] and 
                  delta_timestamp<timeframe and (new_df.iloc[j]['reply'] == new_df.iloc[i]['reply'] 
                                                 or math.isnan(new_df.iloc[j]['reply']))):
                text = str(new_df.iloc[i]['text']) + " " + str(new_df.iloc[j]['text'])
                new_df.at[i, 'text'] = text
                new_df.at[i,'reactions'] = new_df.iloc[j]['reactions'] + new_df.iloc[i]['reactions']
                
                new_df.drop(j, inplace=True)
                new_df.reset_index(drop=True, inplace=True)
                k = k-1
        i=i+1
    return new_df