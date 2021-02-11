This package provides a method to intelligently define the various user-facing processes
that an organization offers to their customers via various channels. For example, a large financial customer
may use this package to define the steps that comprise the various actions a customer may take on their website,
such as paying a bill or transferring funds.

To begin, an activity log for a specific channel must be obtained. For large organizations, a single day of traffic
may suffice. Specifically, the activity log must contain identifiers of the user or user session, 
the activities performed, and corresponding timestamp of each activity.

The package provides functions to convert the activity dataset to a specific class, sequence the activities, fit a
word2vec skipgrams, and lastly cluster the distinct activities based on how often they occur together across the user sessions.