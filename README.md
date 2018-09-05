# VERAM:View-Enhanced-Recurrent-Attention-Model-for-3D-Shape-Classification
Songle Chen, Lintao Zheng, Yan Zhang, Zhixin Sun, Kai Xu*

## Introduction
This code is a Torch implementation of VERAM, a recurrent attention model capable of actively selecting a sequence of views for highly accurate 3D shape classification. VERAM addresses an important issue commonly found in existing attention-based models, i.e., the unbalanced training of the subnetworks corresponding to next view estimation and shape classification. Details of the work can be found [here](http://kevinkaixu.net/projects/veram.html).
![prediction example](https://github.com/charlesq34/3dcnn.torch/blob/master/doc/teaser.png)

## Citation
If you find our work useful in your research, please consider citing:	

	@ARTICLE{8444765, 
        author={S. Chen and L. Zheng and Y. Zhang and Z. Sun and K. Xu}, 
        journal={IEEE Transactions on Visualization and Computer Graphics}, 
        title={VERAM: View-Enhanced Recurrent Attention Model for 3D Shape Classification}, 
        year={2018}, 
        volume={}, 
        number={}, 
        pages={1-1}, 
        doi={10.1109/TVCG.2018.2866793}, 
        ISSN={1077-2626}, 
        month={},}

## Requirements
 1. This code is written in lua and requires [Torch](http://torch.ch/). You should setup torch environment.

 2. if you'd like to train on GPU/CUDA, you have to get the cutorch and cunn packages:
 ```
     $ luarocks install cutorch	
     $ luarocks install cunn
 ```
 
 3. Install other torch packages (nn, dpnn, rnn, image, etc.): ``` $ ./scripts/dependencies_install.sh ```

## Usage 
### Data 
To Train a VERAM model, 3 data files need to be prepared according to the instructions in the fold 'AlexNetFC6Extract', namely:
```
    (1) The visual features for each view of shapes in the training set
    (2) The viusal features for each view of shapes in the testing set
    (3) the confidence for each view of shapes in the training set.
```
The format of the data is hdf5 and each shape category is saved in the dataset 'data\i', i is the category index. The data structure is 'number of shapes in the this category'x12x12x4096, and the data is loaded by imageset2DwithConf.lua.

We have provided the prepared the training and testing data of ModelNet10, the testing data of ModelNet40 in the fold data, they can be used directly.
    

### Train 
To train a VERAM model to classify 3D shapes in ModelNet10 or ModelNet40, please download data files according to the instrction in fold 'data', and then run:
```
  (1) $ th ModelNet10AlexTrain.lua 
  (2) $ th ModelNet10AlexTrain.lua
```
 
 
### Evaluation
The trained VERAM models with 3, 6, 9 views respectively can be downloaded according to the instruction in fold 'model'. To evulate the trained model, just run the lua files in fold 'evaluation', for example:
 ```
     $ th evaluateModelNet40-AlexNet-9Views-0.937196InstanceLevelAccuracy.lua 
```
You can see the details of the classification result, including the predicated category and probability of each shape, the total number, correctly classified number and accuracy of each category.

Taking rendered grayscale image as input and AlexNet as CNN architecture, without applying any data augmentation or network ensemble strategy, the classification results VERAM achieved on ModelNet40 test data are listed below.

|   Accuracy     |  3 views  |  6 views  |  9 views  |
| -------------- |:---------:|:---------:|:---------:|
| instance-level | 92.3825%  | 93.3144%  |  93.7196% | 
|   class-level  | 90.8087%  | 91.6291%  |  92.1244% |


## Acknowledgement
Torch implementation in this repository is based on the code from Nicholas Leonard's recurrent model of visual attention, which is a clean and nice GitHub repo using Torch.
