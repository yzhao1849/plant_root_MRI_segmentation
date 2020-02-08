'prediction_runner.py' is the code for doing segmentation with trained networks.

The command line arguments for 'prediction_runner.py' include:  (Specified in prediction_utils.py, function 'parse_args')
  1. '-nc', '--no-cuda': if True, disables CUDA training. Default is False.
  2. '-cd', '--cuda-device': index of cuda device to use, default=0. Can be a list of indices separated by a whitespace.
  3. '-idd', '--input_data_dir': load data from this directory. Default is '', when a loaded image is used directly
  4. '-od', '--output_dir': required argument, the output will be stored in this directory.
  5. '-md', '--model_dir': required argument, the directory where the trained model is stored. There should be one '.tm' file of a saved model and one 'constants.txt' file in this directory.
  6. '-bs', '--batch_size': required argument, the batch size used for inputting crops to the network (depending on GPU memory. batch size only affects the speed not the result)
All directories should be absolute paths of the directories.

The data in input_data_dir should be of data type uint8, and its shape should be (dim_z, 1, dim_x, dim_y)

The output saved will be of data type uint8, with shape (dim_z, dim_x, dim_y)

