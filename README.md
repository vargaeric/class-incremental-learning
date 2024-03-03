# Master Thesis Progress Tracking

## Week 1 (02.26 - 03.03)

We have changed the class-incremental learning framework that we have used for 
experiments until now, as it did not contain the FOSTER method, which we will 
most likely focus on if we cannot achieve the desired results with the modification
of the iCaRL approach. Regarding the iCaRL approach, we have deprecated the 
modification we made in the previous semester, in which we assigned weights to each
of the exemplars based on Pearson’s correlation coefficient and used those weights
to calculate each of the representations of exemplars.

In our new approach, we aim to modify the iCaRL method in a way to choose exemplars
that best represent the distribution of the data within each class, potentially 
offering a more diverse and representative set of exemplars. To do this, to select
exemplars, we have chosen to use the clustering algorithm k-means. The results can
be seen in the 'results.txt' file but can be summarized with the following lines:

**Original iCaRL method accuracy for 3 tasks: [29.7, 16.05, 14.97]**

**K Means iCaRL method accuracy for 3 tasks: [29.7, 16.2, 15.07]**

As we can see, our approach not only achieved the original iCaRL performance but 
even surpassed it by some decimals (0.15, respectively 0.10). We will continue 
trying to improve this approach while also trying new ones.

**Notes: As no external code is allowed to be used in the master thesis, I have to 
write my own code for iCaRL. The beginning of this can be seen in the 
"class-incremental-learning" folder.**
