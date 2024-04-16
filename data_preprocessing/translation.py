from tqdm import tqdm
def translate_messages(df_original):
    import translators as ts
    translation_text_google = []
    df_translated = df_original.copy()
    for i in tqdm(range(len(df_translated))):
            try:
                message_google = ts.translate_text(query_text = df_translated['text'].iloc[i], translator='google', from_language='auto', to_language='en')
                translation_text_google.append(message_google)
            except:
                translation_text_google.append(df_translated['text'].iloc[i])                 

    df_translated['text'] = translation_text_google
    return df_translated