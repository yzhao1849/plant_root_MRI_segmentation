# plant_root_MRI_segmentation
This repository contains scripts to train 3D UNet and evaluate the performance of trained networks.

The data used for training are 3D MRI image crops, which can be from real MRI images or synthetic images.

The 'models' directory contains python files that define different model structures. The 'jupyter_notebooks' directory 
contains notebooks with different functions. 
Sub directory 'run_predictions' is used for using trained models to produce segmentations.
Sub directory 'datalists' contains npz files of frequently used datalists.

### 1. Location of data and datalists
The data directory is currently located in cuda4:/home/user/zhaoy/local_dir/data.

Datalists are 2D arrays that are used to specify which data to be loaded from the data directory,
and they are used as input arguments for the training script Network_Runner.py.
For more details, please check the README.md in the sub folder 'datalists'.

### 2. Training
The main function for training is defined in Network_Runner.py. The hyperparameters for the network and other 
arguments for the training process are added as command line arguments when running the training script. The whole 
list of command line arguments and their definitions can be seen in Utils/Utils.py in function parse_args(). 

An example of a training command: 

`CUDA_VISIBLE_DEVICES=0 python3 Network_Runner.py -cd 0 -n UNet_3D_noPadding_BNbeforeReLU.py -csl 60
-bs 16 -gi 2 -id 14_01_20_3DUNet -rs -lri 0.0001 -ntcs 60 60 60 -ntbs 40 -vwrc -tts 100 -vi 3000 
-vai 100 -tl 50000 -vl 2000 -tcl 50000 -vcl 2000 
-datad 'directory_of_the_data'
-dd 'datalist_npz_path_of_the_already_generated_noisy_root_images' 
-ddc 'datalist_npz_path_of_the_random_roots_used_in_combining' 
-sdd 'datalist_npz_path_of_the_soil_data_used_in_combining' 
-mod 'path_to_the_model_output_directory'`

This command will train a 3D UNet with 60\*60\*60 image crops (`-csl`), with a batch size of 16(`-bs`)\*2(`-gi`). 
The suffix of the model output directory will be 14_01_20_3DUNet (`-id`). 
Random scaling will be applied on the image crops before inputting to the network (`-rs`).
Learning rate for the optimizer is 0.0001 (`-lri`).
The size of validation crop is also 60\*60\*60 (`-ntcs`). 
The validation batch size is 40 (`-ntbs`).
The validation will be done on random crops from the validation data (`-vwrc`).
If not terminated manually, the training process will run 100 epoches (`-tts`).
Every 3000 training batches, the segmentation output of the test data will be saved (`-vi`).
Every 100 training batches, the validation will be performed (`-vai`).
In every training epoch, 50000 random training crops will be sampled from the noisy dataset (`-tl`), 
and another 50000 random training crops will be sampled from the combined dataset (`-tcl`).
In each round of validation, 2000 random crops from the noisy dataset (`-vl`) and 2000 from the combined dataset (-vcl) are used.
The data will be loaded from the directory specified by `-datad`.
The datalist npz files are used to load the corresponding data, from the noisy dataset (`-dd`), 
or the random root dataset (`-ddc`), or the pure soil data (`-sdd`).
The model outputs including the saved model and saved test segmentation outputs will be stored in `-mod`.

#### More training hyperparameters

- **Training with a customized root weight**

  Need to specify the root weight command line argument, for example: `-rw 10`

- **Training with don't-care flag around the root-soil border**

  Need to set the dont_care argment to True by adding: `-dc`

- **Training with higher loss near edges**

  Set the calculate_gradient_diff_loss argment to True by adding: `-cgdl`

- **Training with importance sampling**

  Set the importance_sampling_root_perc and reweight_loss argments to True by adding: `-isrp -rwl`, 
  and specify an importance offset: `-cwo 0.02`. 
  The importance offset is used to add a minimal sampling probability to all potential image crops.
  Additionally, the weight_base_dir argument needs to be set: 
  `-wbd 'some_directory_where_the_calculated_importance_matrices_will_be_stored'`.
  
- **Training with an earlier plus a later time point**

  Set the use_later_time argment to True by adding: `-ult`,
  and the `-ddc` here should be the datalist of the earlier time point 
  (and the corresponding later time point will be automatically loaded when preparing the training input).
  Also, because time series data only exists in the combined dataset, `-dd` is not needed anymore, 
  and both `-tl` and `-vl` should be set to 0. 
  Moreover, the network should be set to `-n UNet_3D_noPadding_BNbeforeReLU_addOneNewChannel.py` to accept the additional time channel.

- **Training with additional location-dependent info channel**

  - **Info type 1: Distance to pot central axis**

    Set the use_dist_to_center argment to True by adding: `-udtc`. 
    The network should be set to `-n UNet_3D_noPadding_BNbeforeReLU_addOneNewChannel.py` to accept the additional input channel.
  
  - **Info type 2: Depth**

    Set the use_depth argment to True by adding: `-ud`
    The network should be set to `-n UNet_3D_noPadding_BNbeforeReLU_addOneNewChannel.py` to accept the additional input channel.


### 3. Visualization of the training and validation processes
Visualization of the training and validation is done using tensorboard. 
During training, the training loss, validation loss and validation F1 scores are stored in the 'runs' sub-directory of the model output directory,
and one can run command `tensorboard --logdir=runs` to launch tensorboard (need to be installed first) and view the recorded curves.

### 4. Using trained networks
The sub folder 'run_prediction' of this directory is for using trained networks to segment new data, 
please find more details of how to do it in the README.md in 'run_prediction'.

### 5. Evaluating network performance
The jupyter notebook 'evaluate_network_performance.ipynb' in sub folder 'jupyter_notebooks' is used to evaluate
the performance of the models. By following the steps in the notebook,
one can test a trained model on test data, and calculate the distance tolerant F1 scores of the segmentation results.
Further details can be found in the notebook.


