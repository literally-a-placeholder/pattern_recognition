## Instructions

### 1. Clone the DeepDiva 
Clone the project to a new directory using the following command from within this directory: 

```
git clone https://github.com/DIVA-DIA/DeepDIVA.git
cd DeepDiva

```

### 2. Setup Conda Environment
Make sure you have conda installed. Then go to the DeepDiva Directory you cloned in the previous step and run the following commands: 

```
bash setup_environment.sh
source ~/.bashrc
source activate deepdiva
```

### 3. Verify Functionality
Download an MNIST dataset and perform the sample CNN on it to check if everything works as expected: 
```
python util/data/get_a_dataset.py --dataset mnist --output-folder toy_dataset
python template/RunMe.py --output-folder log --dataset-folder toy_dataset/MNIST --lr 0.1 --ignoregit --no-cuda
```

### 4. Import Custom Model
Copy paste our PR_CNN.py file to the ~/DeepDiva/models folder. Open the __init__.py file in the models folder and add the following line below the other imports: 
```
from .PR_CNN import PR_CNN
```

### 5. Import and prepare MNIST dataset
Copy & paste the extracted "mnist-png-format.zip" from ilias to the ~/DeepDiva/ folder. Make sure that the mnist/train subfolder does not contain a file/link called "val"


### 6. Run CNN
Run the CNN using the following command:
```
 python template/RunMe.py --output-folder log --dataset-folder mnist --lr 0.1 --model-name PR_CNN --ignoregit --no-cuda
```

### 7. Frequent Errors and Fixes

#### 7.1 SyntaxError async=True
In case you run into SyntaxErrors with .cuda(async=True), then simply replace all occurrences of .cuda(async=True) with .cuda()

#### 7.2 ModulesNotFound
In case you run into ModulesNotFound Errors, install the missing modules using and retry the command above: 
```
pip install <module_name>
```
#### 7.3 TypeError: can't convert np.ndarray of type numpy.object_.
This error could occur when running the command from step 6. Repeat the environment setup in step 2 and retry running the command from step 6. 

### 8. Visualize Results
Install Tensorflow (includes TensorBoard) or only TensorBoard. Then run the visualization from within the results directory: 
```
cd log/<name of experiment>/mnist/model_name\=PR_CNN/lr\=0.1/no_cuda\=True/<date>
tensorboard --logdir ./ --port 6006
```

