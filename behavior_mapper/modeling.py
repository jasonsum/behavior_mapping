# Package imports
import numpy as np
import pandas as pd
from nltk.tokenize import WhitespaceTokenizer
from gensim.models import word2vec
import collections
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE


def skip_grams (sequence_df,
               feature_size,
               window = 3,
               min_activity_count = 0,
               **kwargs):

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
                                     sg=1, **kwargs) #,sample=1e-5, iter=50)

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
                  min_activity_count = 0,
                  **kwargs):
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
    w2v_dict = skip_grams (sequence_df, 
                          feature_size = feature_size, 
                          window = window, 
                          min_activity_count = min_activity_count,
                          **kwargs)
    
    # Re-map activity IDs to original activity names using mapping dictionary
    activities_features = merge_dicts (activity_map, w2v_dict)
    
    return activities_features

def dbscan_cluster (activities_features,
                    min_samples = 2,
                    eps = .5,
                    **kwargs):
    
    """Performs scikit-learn's DBSCAN clustering on activity-feature dictionary or dataframe
    
    Parameters
    ----------
    activities_features : dictionary (or dataframe of features)
                          Dictionary of activities and corresponding features from word2vec skip grams model 
    min_samples : integer (default=2)
                  Number of samples in a neighborhood for a point to be considered as a core point
    eps : float (default=n.5)
              Maximum distance between two samples for one to be considered as in the neighborhood of the other
    
    Returns
    -------
    pandas dataframe of activities, skipgrams features, and cluster label from DBSCAN
    """

    if type(activities_features) == dict:
      # Create dataframe from activity features dictionary 
      activity_cluster_df = pd.DataFrame.from_dict(activities_features, orient='index')
    
    elif type(activities_features) == pd.core.frame.DataFrame:
      activity_cluster_df = activity_features.copy()

    # Instantiate and fit DBSCAN clustering
    clustering = DBSCAN(min_samples = min_samples, eps = eps, **kwargs).fit(activity_cluster_df)
    # Create and return dataframe of activity name, features, and cluster label
    activity_cluster_df['cluster'] = clustering.labels_
    
    return activity_cluster_df

def dim_reduction (activity_cluster_df):
    """Performs scikit-learn's TSNE to reduce feature dimensionality to 2

    Parameters
    ----------
    activity_cluster_df : dataframe
                          Pandas dataframe of activities, skipgrams features, and cluster label from DBSCAN

    Returns
    -------
    pandas dataframe of activities, skipgrams features, cluster label from DBSCAN, x-value, and y-value
    """

    # Instantiate and fit TSNE
    dim_reduction = TSNE(n_components = 2, random_state = 0, n_iter = 1000, perplexity = 2)
    np.set_printoptions(suppress=True)
    t_dimensions = dim_reduction.fit_transform(activity_cluster_df.drop('cluster', inplace=False, axis=1))

    # Add x and y coordinate values to activity dataframe
    activity_cluster_df['x'] = t_dimensions[:,0]
    activity_cluster_df['y'] = t_dimensions[:,1]

    return activity_cluster_df