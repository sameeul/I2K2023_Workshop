# I2K 2023 Workshop

## Introduction
This repository contains code and sample data for the workshop of [Nyxus](https://github.com/PolusAI/nyxus), a high-performance, scalable image extration library. In this workshop, we will be exploring some basic features of Nyxus Python API using the [EVICAN](https://academic.oup.com/bioinformatics/article/36/12/3863/5814923) dataset. For the purpose of demonestration, a subset of this dataset is converted to OME TIFF format and placed in the `data` directory.

## Getting Started:
We will start with creating a virtual environment in `conda` and install `nyxus`.
```
conda create -n i2k_workshop python=3.9
conda activate i2k_workshop
conda install nyxus -c conda-forge
```

### Dependencies
The following  packages are needed for the workshop.
* bfio ==2.1.9
* Jupyter Notebook
* Matplotlib
* Vaex

`pip` is the preferred way to install these packages.

```
pip install bfio jupyter matplotlib vaex
```

### Cloning The Repository
Now, let's clone the repository and get to the workshop. The `data` folder contains the relavent images for analysis and `workshop.ipynb` notebook file contains all the steps that we will be following. You can either open that notebook in Jupyter or create a new notebook and write the code as we work through the tasks.
```
git clone https://github.com/sameeul/I2K2023_Workshop.git
```
# Tasks
## 1. Inspect intensity and segmentation images
Let's first start with inspecting a pair of intensity and segmentation images before using `Nyxus` to compute various feature values.
```
# inspect intensity image
from bfio import BioReader
import matplotlib.pyplot as plt

with BioReader("./data/evican/intensity/eval_100_SKBR_ch2.ome.tif") as br:
    print(f"Image Shape {br.shape}")
    plt.imshow(br[:].squeeze())

with BioReader("./data/evican/label/eval_100_SKBR_ch2.ome.tif") as br:
    print(f"Image Shape {br.shape}")
    plt.imshow(br[:].squeeze())
```

## 2. Calculate a single feature of one image pair
Nyxus can be instantiated with a list of image pairs and and feature list to calculate the feature values. To start, we will just inspect one single feature. 
```
from nyxus import Nyxus
int_image_path = ["./data/evican/intensity/eval_100_SKBR_ch2.ome.tif"]         
seg_image_path = ["./data/evican/label/eval_100_SKBR_ch2.ome.tif"]
nyx = Nyxus(["MEAN"])
feature_vals = nyx.featurize_files(int_image_path, seg_image_path)
```
A full list of all the available features can be found [here](https://nyxus.readthedocs.io/en/latest/featurelist.html)

## 3. Calculate a bunch of features for one image pair
We can also pass a list of features in a list during `Nyxus` instantiation to calculate them all together.
```
nyx = Nyxus(["MIN", "MAX", "MEAN", "CIRCULARITY"])
```

## 4. Calculate a group of features
For convenience, apart from defining feature set by explicitly specifying comma-separated feature code, `Nyxus` lets a user specify popular feature groups. Supported feature groups are:

------------------------------------
| Group code | Belonging features |
|--------------------|-------------|
| \*all_intensity\* | integrated_intensity, mean, median, min, max, range, standard_deviation, standard_error, uniformity, skewness, kurtosis, hyperskewness, hyperflatness, mean_absolute_deviation, energy, root_mean_squared, entropy, mode, uniformity, p01, p10, p25, p75, p90, p99, interquartile_range, robust_mean_absolute_deviation, mass_displacement
| \*all_morphology\* | area_pixels_count, area_um2, centroid_x, centroid_y, weighted_centroid_y, weighted_centroid_x, compactness, bbox_ymin, bbox_xmin, bbox_height, bbox_width, major_axis_length, minor_axis_length, eccentricity, orientation, num_neighbors, extent, aspect_ratio, equivalent_diameter, convex_hull_area, solidity, perimeter, edge_mean_intensity, edge_stddev_intensity, edge_max_intensity, edge_min_intensity, circularity
| \*basic_morphology\* | area_pixels_count, area_um2, centroid_x, centroid_y, bbox_ymin, bbox_xmin, bbox_height, bbox_width
| \*all_glcm\* | glcm_asm, glcm_acor, glcm_cluprom, glcm_clushade, glcm_clutend, glcm_contrast, glcm_correlation, glcm_difave, glcm_difentro, glcm_difvar, glcm_dis, glcm_energy, glcm_entropy, glcm_hom1, glcm_hom2, glcm_id, glcm_idn, glcm_idm, glcm_idmn, glcm_infomeas1, glcm_infomeas2, glcm_iv, glcm_jave, glcm_je, glcm_jmax, glcm_jvar, glcm_sumaverage, glcm_sumentropy, glcm_sumvariance, glcm_variance
| \*all_glrlm\* | glrlm_sre, glrlm_lre, glrlm_gln, glrlm_glnn, glrlm_rln, glrlm_rlnn, glrlm_rp, glrlm_glv, glrlm_rv, glrlm_re, glrlm_lglre, glrlm_hglre, glrlm_srlgle, glrlm_srhgle, glrlm_lrlgle, glrlm_lrhgle
| \*all_glszm\* | glszm_sae, glszm_lae, glszm_gln, glszm_glnn, glszm_szn, glszm_sznn, glszm_zp, glszm_glv, glszm_zv, glszm_ze, glszm_lglze, glszm_hglze, glszm_salgle, glszm_sahgle, glszm_lalgle, glszm_lahgle
| \*all_gldm\* | gldm_sde, gldm_lde, gldm_gln, gldm_dn, gldm_dnn, gldm_glv, gldm_dv, gldm_de, gldm_lgle, gldm_hgle, gldm_sdlgle, gldm_sdhgle, gldm_ldlgle, gldm_ldhgle
| \*all_ngtdm\* | ngtdm_coarseness, ngtdm_contrast, ngtdm_busyness, ngtdm_complexity, ngtdm_strength
| \*all_easy\* | All the features except the most time-consuming GABOR, GLCM, and the group of 2D moment features
| \*all\* | All the features 

We can calculate all the intensity features by simply doing this.
```
nyx = Nyxus(["*ALL_INTENSITY*"])
```

## 5. Calculate multiple feature groups of one image pair
Multiple feature groups codes can also be passed as a list to compute them in one pass by doing the following.
```
nyx = Nyxus(["*ALL_INTENSITY*", "*basic_morphology*"])
```

## 6. Calculate features of all the images from one directory 
Similar to a single image pair, `Nyxus` can be used to calculate features from all the images from one directory using `featurize_directory` method. 
```
from nyxus import Nyxus
int_image_dir_path = "./data/evican/intensity"         
seg_image_dir_path = "./data/evican/label"
nyx = Nyxus(["MEAN"])
feature_vals = nyx.featurize_directory(int_image_dir_path, seg_image_dir_path)
```

## 7. Calculate features from image pair as 2D numpy array
Nyxus also has an API to ingest image data as 2D/3D numpy array and calculate feature values. This is useful when the source data is not in one of the natively supported formats. To simulate this we will first read the image pair using `bfio` and store them in 2D `numpy` array. 
```
from bfio import BioReader
import numpy as np

int_image_path = "./data/evican/intensity/eval_100_SKBR_ch2.ome.tif"         
seg_image_path = "./data/evican/label/eval_100_SKBR_ch2.ome.tif"

# load images as 2D numpy array
int_img = BioReader(int_image_path)
seg_img = BioReader(seg_image_path)
int_image_data = int_img[:].squeeze()
print(int_image_data.shape)
seg_image_data = seg_img[:].squeeze()
print(seg_image_data.shape)
int_img.close()
seg_img.close()
```
Now, we can use the `featurize` method to calculate the feature values.
```
feature_vals = nyx.featurize(int_image_data, seg_image_data)
```

## 8. Save output to arrow
By default, feature values are returned as a `pandas` dataframe. However, `Nyxus` output can be also saved as `Arrow` files, both `IPC` and `Parquet` format.

```
from nyxus import Nyxus
int_image_path = ["./data/evican/intensity/eval_100_SKBR_ch2.ome.tif"]         
seg_image_path = ["./data/evican/label/eval_100_SKBR_ch2.ome.tif"]
nyx = Nyxus(["*ALL_INTENSITY*", "*basic_morphology*"])
nyx.create_arrow_file()
feature_vals = nyx.featurize_files(int_image_path, seg_image_path)
```