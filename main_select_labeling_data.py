from data_labeling.select_for_labeling import select_data_for_labeling
import pandas as pd

raw_data = pd.read_csv("./data/trial.csv")
df = select_data_for_labeling(raw_data)
df.to_csv("./data/labeling_data.csv")