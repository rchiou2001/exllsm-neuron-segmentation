# Neuron Segmentation Pipeline

This is an adaptation of Linus Meienberg [tool](https://github.com/randomstructures/ExLSM-Image-Segmentation) for segmenting large scale volumetric microscopy images to be used as part of [Expansion Microscopy Pipeline](https://github.com/JaneliaSciComp/expansion-microscopy-pipeline) for automatic neuron segmentation.


## Quick Start

To run the tool standalone first setup a conda environment using the following command:
```
conda create -n neuron-segmentation python=3.9
conda env update -n neuron-segmentation -f conda-requirements.yml
```
then activate it:
`
conda activate neuron-segmentation
`

Below is an example of how to volume segmentation:
```
python volumeSegmentation.py \
    -i /test/ExM/P1_pIP10/20200808/images/export_substack_crop.n5 \
    -id /c1/s0 \
    -o /results/test/Q1seg.n5 \
    -od /segmented/s0 \
    -m /model_location/trained_models/neuron4_p2/neuron4_150.h5 \
    --image_shape 10325,6150,2560 \
    --start 0,0,0 \
    --end 250,250,250 \
    --set_gpu_mem_growth \
    --unet_batch_size 1 \
    --model_input_shape 220,220,220 \
    --model_output_shape 132,132,132 \
    --with_post_processing \
    --high_threshold 0.98 \
    --low_threshold 0.2 \
    --small_region_probability_threshold 0.9 \
    --small_region_size_threshold 1000
```

Segmenting the volume can be preceded by calculating the global scaling factor using:
```
python volumeScalingFactor.py \
    -i /test/ExM/P1_pIP10/20200808/images/export_substack_crop.n5 \
    -d /c1/s0 \
    -p 0.1 \
    --partition_size 396,396,396 \
    --start 0,0,0 \
    --end 250,250,250
```

## Model training and evaluation

### Dataset preparation

Tool for peparing training and evaluation datasets. 

This tool prepares HDF5 datsets (Regions) composed of a raw data and ground truth data channels stored in different groups for model training and evaluation. The Regions are subdivided into a grid of potential Training Examples. Because much of the 3D volume may be empty space, a threshold is set to identify Training Examples with signal. An above-threshold-ratio is set to determine the percentage of Training Examples that should be above the threshold (i.e. Training Examples with and without signal).

Regions = crops that are manually extracted from a image volume; stored at HDF5 datasets where channels are in different groups
Training examples = small crops automatically extracted from Regions using this script

Usage: Usage: 

     python datasetPreparation_cmd.py --region-dir /REGION-DIR/ --dataset-dir /DATASET-DIR/ --output-dir /DATASET-DIR/ --above-threshold-ratio 0.9 --samples-per-region 50

| Argument   | Default | Description                                                                           |
|------------|---------|---------------------------------------------------------------------------------------|
| --output-dir |  | output directory |
| --dataset-dir |  | directory where output HDF5 dataset is to be saved |
| --region-dir |  | directory where input HDF5 datasets are located |
| --above-threshold-ratio | 0.9 | ratio of Training Examples above the set threshold versus training examples below the set threshold to be included |
| --samples-per-region | 50 | number of Training Examples per Region to include in the dataset |

For preparing the dataset, training and evaluating the model there are ways to use the tools. One is to invoke the tool from the command line using the corresponding command ending in '_cmd.py': `datasetPreparation_cmd.py`, `training_cmd.py` and/or `evaluation_cmd.py` and the second one is to use the tool without the '_cmd' suffix from a Jupyter notebook.
