# VERAM:View-Enhanced-Recurrent-Attention-Model-for-3D-Shape-Classification
Songle Chen, Lintao Zheng, Yan Zhang, Zhixin Sun, Kai Xu*

## Introduction
This code is a Torch implementation of VERAM, a recurrent attention model capable of actively selecting a sequence of views for highly accurate 3D shape classification. VERAM addresses an important issue commonly found in existing attention-based models, i.e., the unbalanced training of the subnetworks corresponding to next view estimation and shape classification. Details of the work can be found [here](http://kevinkaixu.net/projects/veram.html).

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
To Train a VERAM model, 3 data files need to be prepared according to the instructions in the fold AlexNetFC6Extract, namely:

    (1) The visual features for each view of shapes in the training set
    (2) The viusal features for each view of shapes in the testing set
    (3) the confidence for each view of shapes in the training set.

The format of the data is hdf5 and each shape category is saved in the section 'data\i', i is the category index. The data structure is 'number of shapes in the this category'x12x12x4096, and the data is read by imageset2DwithConf.lua.

We have provided the prepared the training and testing data of ModelNet10, the testing data of ModelNet40 in the fold data, they can be used directly.
    

### Train 
To train a MV-RNN model to classify object for root node:
```
$ th train.lua 
```
 Run `th train.lua -h` to see additional command line options that may be specified.
 
 If you want to train hierarchy MV-RNN models for every node of all classes, run:
   ./scripts/train_hierarchy_mvrnn.sh
 
### Evaluation
We have trained all models(MV-RNN models) for every node of class chair(subclass1), you can see the evaluation results following opeartions below.
To evulate the MV-RNN model for the root node:
 ```
$ th eval_demo.lua 
```
You can see retrive examples by running:
```
$ th retrive_demo.lua
```
the results are saved in the folder `retrive_res`.
(note: if encounter an error due to ViewSelect.lua, you can fix it by uncommenting the line 35 in ViewSelect.lua)
<br>

### Example output by retrive_demo.lua
<br>
 1. Example of ten views comparision betwwen input and retrive data 
<br>
 <p>   first row for input data, second row for retrive data <p/>
 (1) <br><img src="images/compare1.png" width="80%"><br>  
 (2) <br><img src="images/compare2.png" width="80%">

<br>
 2. Example of view sequence
<br>
 (1)<br><img src="images/view_seqs1.png" width="60%"><br> 
 (2)<br><img src="images/view_seqs2.png" width="60%">

## Acknowledgement
Torch implementation in this repository is based on the code from Nicholas Leonard's recurrent model of visual attention, which is a clean and nice GitHub repo using Torch.
