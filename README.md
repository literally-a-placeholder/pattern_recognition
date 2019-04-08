# Pattern Recognition Group Task
Group name: literally_a_placeholder
Members: Tim Ogi, Mario Moser, Michael Baur, Dominik Meise, Timo Spring

## Results
SVM Results: [results.txt](https://github.com/literally-a-placeholder/pattern_recognition/blob/master/SVM/svm_results.txt), [stdout.txt](https://github.com/literally-a-placeholder/pattern_recognition/blob/master/SVM/stdout.txt)

MLP Results: [MLP](https://github.com/literally-a-placeholder/pattern_recognition/tree/master/MLP)

CNN Results: [CNN](https://github.com/literally-a-placeholder/pattern_recognition/tree/master/CNN/results)


## Run
Clone the project and run the following command from the root folder to install all dependencies. 

```python
pip install -r requirements.txt
```

Move the mnist_train.csv and mnist_train.csv files inside the dataset folder. 
Then you are ready to rumble!


## Install new Packages
Make sure to install new packages using the following commands in order to make sure that the dependencies are listed in the requirements.txt file: 

```python
pip install <package> 
pip freeze > requirements.txt
```

## Visualize CNN Results
Results for the CNN Task can be found in ~/CNN/results. 
It also contains the tensorboard files that can be viewed using the following command: 
```
cd results/pr_cnn/mnist/model_name\=PR_CNN/
tensorboard --logdir ./ --port 6006
```
Have fun!

