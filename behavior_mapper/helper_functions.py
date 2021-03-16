# Package imports
import numpy as np
import pandas as pd


def csv_import (file,
                ID,
                activity,
                timestamp_col):
    """Imports csv file and creates dataframe indicating the columns corresponding to id, activity, and occurrence time
    
    Parameters
    ----------
    file : string
           File path of csv
    ID : string
         Column featuring merge key to match session activities
    activity : string
               Column featuring user activity
    timestamp_col : string
                    Column featuring datetime of activities
    
    Returns
    -------
    pandas dataframe of merging ID, activity, and timestamp
    """
    
    input_df = pd.read_csv(file,
                           header=0,
                           usecols=[ID,activity,timestamp_col],
                           dtype={ID:'string', activity:'string'},
                           parse_dates = [timestamp_col])
    return input_df

