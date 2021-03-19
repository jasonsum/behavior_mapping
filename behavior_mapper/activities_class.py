# Package imports
import numpy as np
import pandas as pd
from itertools import groupby

class activities(pd.DataFrame):                                                                                                                                             
    def __init__(self, *args, **kwargs):                                                                                                                                   
        kwargs['columns'] = ['ID', 'activity', 'occurrence', 'activity_ID']                                                                                                                        
        super(activities, self).__init__(*args, **kwargs) 
    
    # Method ensures that other methods return 
    # instance of custom class instead of regular DataFrame
    @property
    def _constructor(self):
        return activities
    
    def remove_activities (self,
                          drop_activities):
        
        """Removes rows of activities that contain a string pattern provided in drop_activities

        Parameters
        ----------
        drop_activities : list of strings
                          Character pattern matches indicating activities to be removed

        Returns
        -------
        activities class datframe with rows removed where string matches of drop_activites are found
        """

        pattern_list = '|'.join(tuple(drop_activities))
        self = self[~self['activity'].str.contains(pattern_list, regex=True)]
        self.reset_index(inplace=True, drop=True)
        return self
    
    def create_dicts (self):

        """Creates lookup dictionary for activity mapping with activity and activity_ID
        and dictionary of activities and their session counts

        Parameters
        ----------

        Returns
        -------
        dictionary of activity and activity_ID 
        dictionary of activity and session counts
        """

        # Create mapping dictionary of activity_ID:activity pairs
        activity_map = self['activity'].drop_duplicates().reset_index(drop=True).to_dict()

        # Create count dictionary of activity:session ID count pairs
        activity_counts = self.groupby('activity')['ID'].nunique().to_dict()


        return activity_map, activity_counts
    
    def map_activities (self,
                       activity_map):
    
        """Populates activity_ID column in activities class dataframe corresponding to activity_map

        Parameters
        ----------
        activity_map: dictionary
                      Dictionary of activity and activity_ID 

        Returns
        -------
        activities class datframe with activity_ID column populated based on corresponding activity_map integer value
        """

        # Reverse key and value pairs of dictionary to map
        dict_reversed = {y:x for x,y in activity_map.items()}
        self['activity_ID'] = self['activity'].map(dict_reversed)
        return self
    
    def sequence (self,
                 min_num,
                 remove_repeats=True):
        """Creates dataframe of sequenced activities according to ID and ascending occurrence

        Parameters
        ----------
        min_num : integer
                  Minimum number of steps to retain sequence
        remove_repeats : boolean (default=True)
                         Determination if consecutive repeats of activities in sequences should be reduced to a single occurrence

        Returns
        -------
        pandas dataframe containing sequences of activity_ID according to ID and ascending occurrence
        """

        self.sort_values('occurrence', ascending=True, inplace=True)

        # Cast activity_ID field as string
        self['activity_ID'] = self['activity_ID'].astype(str)

        # Create a string and list form of activity_IDs group concatenated by ID
        sequences = {'seq_list': self.groupby('ID').apply(lambda x: list(x['activity_ID']))}
        sequence_df = pd.DataFrame(data=sequences)

        # Remove consecutive repeats if remove_repeats=True
        if remove_repeats == True:
            sequence_df['seq_list'] = sequence_df['seq_list'].apply(lambda seq: [x[0] for x in groupby(seq)])

        if remove_repeats == False:
            pass
        
        # Remove sequences below the minimum step threshold
        sequence_df['step_num'] = sequence_df.seq_list.str.len()
        sequence_df = sequence_df[sequence_df['step_num'] >= min_num]

        # Create white space delimited string of event IDs
        sequence_df['seq_str'] = [' '.join(map(str, l)) for l in sequence_df['seq_list']]
        return sequence_df
    
    def create_corpus (self,
                       min_num,
                      drop_activities = None,
                      remove_repeats = True):
        
        """Creates activity mapping dictionary and dataframe of sequences of activity_ID 
        to be tokenized by white spaces and used in word2vec skip grams fitting.

        Parameters
        ----------
        min_num : integer
                  Minimum number of steps to retain sequence
        remove_repeats : boolean (default=True)
                         Determination if consecutive repeats of activities in sequences should be reduced to a single occurrence
        drop_activities : list of strings
                          List of strings corresponding to undesirable patterns

        Returns
        -------
        pandas dataframe containing sequences of activity_ID according to ID and ascending occurrence
        dictionary of activity and activity_ID
        dictionary of activity and session counts
        """
        
        # Remove activities if specified
        if drop_activities != None:
            activities_df = self.remove_activities(drop_activities = drop_activities).copy()
        else:
            activities_df = self.reset_index(drop=True).copy()
        
        # Create activity map dictionary and update activities dataframe to include corresponding activity_ID
        activity_map, activity_counts = activities_df.create_dicts()
        activities_df = activities_df.map_activities(activity_map)
        
        # Create sequences of activities
        sequence_df = activities_df.sequence(min_num = min_num, remove_repeats = remove_repeats)
        return sequence_df, activity_map, activity_counts