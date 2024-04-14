# Master Thesis Progress Tracking

## Week 6 (04.01 - 04.07)

This week, there were no significant changes made to the code because we focused 
more on trying to reproduce the results from the original iCaRL method's paper. 
These attempts mainly involved adjusting the hyperparameters of our model to 
match those specified in the iCaRL paper.

## Week 5 (03.25 - 03.31)

We have implemented two new methods for exemplar selection:

**Median Selection**:
This method selects exemplars based on their proximity to the median of the class 
features instead of the mean. This can be more robust to outliers compared to the 
mean-based selection.

**Density-Based Selection**:
Select exemplars based on a density measure, where exemplars are chosen from dense 
regions of the feature space, assuming that these regions represent the class more 
accurately.

Using these two new methods, we will conduct several experiments to see if we can 
achieve higher accuracies.

## Week 4 (03.18 - 03.24)

We have implemented our own method for exemplar selection using K-means clustering;
however, this function is currently not in use. Furthermore, we have made all the
necessary modifications to ensure the reproducibility of our experiments. This
includes setting the seeds for all the packages we have used.

## Week 3 (03.11 - 03.17)

We have continued to write our own implementation of the iCaRL method. The current 
state of the code theoretically includes all components of iCaRL, but we have
encountered some issues. More precisely, when we use the nearest-mean-of-exemplars, 
the classification is far worse than it is with the base neural network. The 
execution time is also slow, so it could be improved, but it is not a must.

## Week 2 (03.04 - 03.10)

We have continued to write our own implementation of the iCaRL method. The current 
state of the code does not yet contain parts of the iCaRL method, nor does it 
include any other class-incremental learning approach. The code we have written 
reflects how a machine learning model (ResNet18, in our case) would behave if the 
data were to come sequentially. This is the foundation upon which any 
class-incremental learning method can and will be implemented. **The current accuracy
for 5 tasks (each task containing 20 classes) on the CIFAR-100 dataset is as follows:
[36.6, 17.0, 9.83, 9.24, 6.85]. We will use these values as an approximate baseline 
for future implementations and testing.**

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
