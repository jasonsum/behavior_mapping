Behavior Mapping
-----------------------------------------------------------------------------------------------------------------------------------------------------

Within the customer experience ecosystem, there is a a plethora of niche forms of analytics that often revolve around understanding and sometimes even predicting customer behavior. 
Niche offerings, such as journey analytics and real-time orchestration, oftentimes entail deciphering customers' intentions and subsequently driving their action (or inaction).
However, while driving or encouraging customer action is oftentimes the core of product offerings in the space, the biggest barrier for complex organizations resides
in the deciphering of customers' intention. 

In e-commerce, it's relatively easy to determine a customer's intent. If a customer is browsing electric coffee makers on Bed, Bath, and Beyond's website, it does not take a complex
model to decipher that the customer may have an interest in coffee makers, coffee beans, coffee grinders, etc. Most e-commerce platforms make such inferences relatively easy based on browsing history. 

When we look to apply the same logic to say, a nationally-recognized bank's website, it becomes more difficult. Suddenly, it's not as easy to decipher what pages or actions relate to a customer 
attempting to take out a loan or apply for a mortgage. The steps that comprise these long processes are oftentimes scattered across dozens of pages or steps and buried under years of releases.  

Enter, this open-source python package. It provides a method to intelligently distinguish the various user-facing processes that an organization offers to their customers via various channels. For example, a large financial organization may use this package to define the steps that comprise the various actions a customer may take on their website,
such as paying a bill or transferring funds. It can be used for other high-traffic channels as well, such as IVR and mobile app platforms. 

To begin, an transactional log for a specific channel must be obtained. For large organizations, a single day of customer traffic may suffice. Specifically, the activity log must contain columns identifying the user or user session, the activity performed, and the corresponding timestamp of the activity.

The package provides functions to convert the activity dataset, described above, to a custom class, sequence the activities, fit a word2vec skipgrams, and lastly cluster the distinct activities based on how often they occur together across the user sessions. Activities with the same cluster typically happen together, indicating the steps of cohesive processes in the given channel.