Datalists are 2D arrays (compressed and saved in the 'dataset.npz' file in each folder) that are used to specify which data to be loaded from the data directory,
and they are used as input arguments for the training script Network_Runner.py.

In a datalist, each row of the array represents one specific image, and each column is one feature of this image.
The features are in the following order: *radius_multiplier, rotation, x_flip, y_flip, x_y_swap, noise_type, noise_scale, data_name, dim_z, dim_x, dim_y, real_data*

The datalists here are:
- **20190918_oguz_datasets_non_combining_withTrainVal_visReal2:** the synthetic noisy root images in the original dataset
- **20191018_randomVirtualRoots_only_visOguzReal2_earlierTimePoint:** the randomly generated root structures at an earlier time point, used for combining
- **20191018_randomVirtualRoots_only_visOguzReal2_laterTimePoint:** the randomly generated root structures at an later time point, used for combining.
  Each data in it has a corresponding earlier version in 20191018_randomVirtualRoots_only_visOguzReal2_earlierTimePoint
- **20191017_pure_soil_new2:** the real soil data used for combining with the root structures