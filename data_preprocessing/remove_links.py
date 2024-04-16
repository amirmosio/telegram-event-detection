def remove_links(df):
    url_pattern = r'https?://\S+'
    df_original = df.copy()
    df_original = df_original.dropna(subset=['text']) #drop messages composed by csv files
    rows_with_links = df_original[df_original['text'].str.contains(url_pattern)] #drop message composed by links
    df_original = df_original.drop(rows_with_links.index)
    df_original = df_original.reset_index(drop=True) #reset index
    return df_original