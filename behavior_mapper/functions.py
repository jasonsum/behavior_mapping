# Package imports
import numpy as np
import pandas as pd
from nltk.tokenize import WhitespaceTokenizer
from gensim.models import word2vec
import collections
from sklearn.cluster import OPTICS

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

def skip_grams (sequence_df,
               feature_size,
               window = 3,
               min_activity_count = 0):
    """Vectorizes sequences by blank space and returns skip gram features for 
    each activity_ID in sequences
    
    Parameters
    ----------
    sequence_df : dataframe
                  Pandas dataframe containing sequences of activity_ID
    feature_size : integer (default=100)
                   Number of dimensions or size of vector to produce for each activity
    window : integer (default=3)
             Size of context window for each activity
    min_activity_count : integer (default=0)
                         Minimum number of activity instances to be considered
    
    
    Returns
    -------
    dictionary of activity_IDs and corresponding features from word2vec skip grams model
    """
    
    tokenizer = WhitespaceTokenizer()
    tokenized_corpus = [tokenizer.tokenize(sequence) for sequence in sequence_df['seq_str']]

    # Train model on corpus using skip-gram method
    w2v_model = word2vec.Word2Vec(tokenized_corpus, size=feature_size, 
                                     window=window, min_count=min_activity_count,
                                     sample=1e-5, iter=50, sg=1)

    # Get unique list of activities
    vocab_activities = [k for k in w2v_model.wv.vocab.keys()]

    # Zip activity_ID and features from w2v model
    w2v_dict = dict(zip(vocab_activities,w2v_model.wv[vocab_activities]))
    
    return w2v_dict

def merge_dicts (activity_map,
                w2v_dict):
    """Combines activity map with activity-feature dictionary, matching on activity_ID from both 
    
    Parameters
    ----------
    activity_map: dictionary
                  Dictionary of activity and activity_ID
    w2v_dict : dictionary
               Activity-feature dictionary of activities from w2v model
    
    Returns
    -------
    dictionary of activity names and corresponding features from word2vec skip grams model
    """
    
    # Cast keys as integer
    w2v_dict = {int(k):v for k,v in w2v_dict.items()}
    
    # Sort both dictionaries by activity_ID
    activity_map_sorted = collections.OrderedDict(sorted(activity_map.items()))
    w2v_dict_sorted = collections.OrderedDict(sorted(w2v_dict.items()))
    
    # Zip activity names and feature values into new dictionary
    activities_features = dict(zip(activity_map_sorted.values(),w2v_dict_sorted.values()))
    
    return activities_features

def fit_sequences (sequence_df,
                  activity_map,
                  feature_size,
                  window = 3,
                  min_activity_count = 0):
    """Vectorizes sequences by blank space and returns skip gram features for 
    each activity in sequences in dictionary
    
    Parameters
    ----------
    sequence_df : dataframe
                  Pandas dataframe containing sequences of activity_ID
    feature_size : integer (default=100)
                   Number of dimensions or size of vector to produce for each activity
    window : integer (default=3)
             Size of context window for each activity
    min_activity_count : integer (default=0)
                         Minimum number of activity instances to be considered
        
    Returns
    -------
    dictionary of activity_IDs and corresponding features from word2vec skip grams model
    """
    
    # Instantiate and fit word2vec skip grams model
    w2v_dict = skip_grams (sequence_df, feature_size = feature_size, window = window, min_activity_count = min_activity_count)
    
    # Re-map activity IDs to original activity names using mapping dictionary
    activities_features = merge_dicts (activity_map, w2v_dict)
    
    return activities_features

def optics_cluster (activities_features,
                    min_samples = 3,
                    metric = 'euclidean',
                    max_eps = np.inf,
                    cluster_method = 'dbscan',
                    min_cluster_size = 3):
    
    """Performs scikit-learn's OPTICS clustering on activity-feature dictionary 
    
    Parameters
    ----------
    activities_features : dictionary
                          Dictionary of activities and corresponding features from word2vec skip grams model
    min_samples : integer (default=3)
                  Number of samples in a neighborhood for a point to be considered as a core point
    metric : string (default='euclidean')
             Metric to use for distance computation
    max_eps : float (default=np.info)
              Maximum distance between two samples for one to be considered as in the neighborhood of the other
    cluster_method : string (deafult='dbscan')
                     Extraction method used to extract clusters 
    min_cluster_size : integer (default=3)
                       Minimum number of samples in an OPTICS cluster
    
    Returns
    -------
    pandas dataframe of activities, skipgrams features, and cluster label from OPTICS
    """
    # Instantiate and fit OPTICS clustering
    clustering = OPTICS(min_samples = min_samples,
                       metric = metric,
                       max_eps = max_eps,
                       cluster_method = cluster_method,
                       min_cluster_size= min_cluster_size).fit(np.array(list((activities_features.values()))))
    # Create and return dataframe of activity name, features, and cluster label
    activity_cluster_df = pd.DataFrame.from_dict(activities_features, orient='index')
    activity_cluster_df['cluster'] = clustering.labels_
    
    return activity_cluster_df