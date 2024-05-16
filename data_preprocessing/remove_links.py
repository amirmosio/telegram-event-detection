def remove_links_and_empty_messages(df):
    df_original = df.copy()
    df_original = df_original.dropna(subset=['text']) #drop messages composed by csv files

    url_pattern = r'https?://\S+'
    df_original['text'] = df_original['text'].replace(url_pattern, '#link', regex=True)  #replcae links with 

    df_original = df_original.reset_index(drop=True) #reset index
    return df_original