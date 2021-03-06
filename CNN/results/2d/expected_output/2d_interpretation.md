### Results interpretation

The permuted dataset shows slightly less accuracy of around 95-96% compared to the normal mnist of around 97-98%.  

Furthermore, we can see that it takes more epochs until the accuracy stabilizes. For the normal MNIST it starts stabilizing around 25 epochs with a few dips in later epochs, with consistent results past 70 epochs. 

For the permuted dataset, the stabilizing only starts around 30 epochs and continues to stabilize as well after 70 epochs. 

Applying a fixed random permutation to the pixels of the mnist images therefore, makes the classification slightly harder (since pixels are distributed across the image and not similarly grouped as with the normal mnist), but still yields in very good results compared to other approaches. 
