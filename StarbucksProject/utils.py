import pandas as pd
import missingno as msno

def create_offer_id_col(val):
    '''
    function created with the puporse of interact with all values of a column
    and by that differentiate the first word and then separate into a new column

    input-> val: value of a column
    '''
    if list(val.keys())[0] in ['offer id', 'offer_id']:
        return list(val.values())[0]
    
def create_amount_col(val):
    '''
    function created with the puporse of interact with all values of a column
    and by that differentiate the first word and then separate into a new column
    
    input-> val: value of a column
    '''

    if list(val.keys())[0] in ['amount']:
        return list(val.values())[0]

def first_infos(df: pd.DataFrame):
    '''
    function created with the puporse of get a first look in the dataframe selected
    
    input-> df: dataframe that you want to have a first look
    '''
    print(df.info())
    msno.bar(df, figsize=(5,3), fontsize=15)
    print(df.sample(3))

