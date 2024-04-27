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

def sum_numbers_in_string(text):
  """
  This function sums all the numbers present in a string. It can be used to count the reactions in a message.

  Args:
      text: The string containing numbers.

  Returns:
      The sum of all the numbers in the string. If no numbers are found, returns 0.
  """
  total_sum = 0
  current_number = ""
  for char in text:
    if char.isdigit():
      current_number += char
    else:
      if current_number:
        total_sum += int(current_number)
        current_number = ""
  # Add the last number if it exists
  if current_number:
    total_sum += int(current_number)
  return total_sum