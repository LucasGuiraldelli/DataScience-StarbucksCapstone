import pandas as pd
import missingno as msno

def create_offer_id_col(val):
    if list(val.keys())[0] in ['offer id', 'offer_id']:
        return list(val.values())[0]
    
def create_amount_col(val):
    if list(val.keys())[0] in ['amount']:
        return list(val.values())[0]

def first_infos(df: pd.DataFrame):
    print(df.info())
    msno.bar(df, figsize=(5,3), fontsize=15)
    df.sample(3)

