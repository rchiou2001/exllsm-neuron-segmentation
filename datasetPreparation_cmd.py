"""
This script prepares a Dataset of 3D tensors as described in the Dataset3D.py module.
It can be used to train and evaluate machine learning models.
"""

import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from tools.Dataset3D import (Dataset, getRandomIndices,
                             sampleMaskProportion, thresholdedSampling)
from tools.preProcessing import (calculateScalingFactor, scaleImage)
from tools.tilingStrategy import UnetTiler3D


def main():

    parser = argparse.ArgumentParser(description='Dataset preparation')

    parser.add_argument('--output-dir',
                        dest='output_directory', type=str,
                        help='Output directory')

    parser.add_argument('--region-dir',
                        dest='region_directory', type=str,
                        help='Region directory')

    parser.add_argument('--dataset-dir',
                        dest='dataset_path', type=str,
                        help='Dataset directory')

    parser.add_argument('--above-threshold-ratio',
                        dest='above_threshold_ratio', type=float,
                        default=0.9,
                        help='Probability threshold to include a sample')

    parser.add_argument('--samples-per-region',
                        dest='samples_per_region', type=int,
                        default=50,
                        help='How many samples to include per region')

    args = parser.parse_args()

    os.makedirs(args.output_directory, exist_ok=True)

    # A 'specimen' is an entire image dataset that was generated
    # in a microscopy session
    # 'Regions' are large crops that are manually extracted from
    # specimens and show anatomical structures of interest
    # Training examples are small crops automatically extracted from
    # regions by this script.

    # Indicate the storage location of the regions (Here a common base directory is assumed)
    # Here we assume that regions are stored as hdf5 datasets where channels are in different groups

    # Relative paths to the rehions from the base directory
    region_paths = ['A1/A1.h5', 'A2/A2.h5', 'B1/B1.h5']
    # A list of names that should be used as identifiers for the regions
    regions = ['A1', 'A2', 'B1']

    # Retrieve Region data files
    image_paths = [args.region_directory +
                   '/' + region for region in region_paths]
    # open the image regions
    regions_h5 = [h5py.File(ip, mode='r+') for ip in image_paths]

    print('Accessing region files')
    for i, h5 in enumerate(regions_h5):
        print(regions[i])
        print(h5.filename)
        print(h5['t0'].keys())
        print('')

    # For every region the image and the mask channel have to be specified.
    # References to the (large) image and mask arrays are stored in two dictionaries
    # image[region] eg. image['A1'] should point to the image channel of the respective hdf5 file
    image = {}
    # mask[region] should point to the ground truth channel of the hdf5 file
    mask = {}
    # sample entry : image['region'] = regions_h5[0]['t0/channel0']
    image['A1'] = regions_h5[0]['t0/channel0']
    mask['A1'] = regions_h5[0]['t0/channel2']
    image['A2'] = regions_h5[1]['t0/channel0']
    mask['A2'] = regions_h5[1]['t0/channel2']
    image['B1'] = regions_h5[2]['t0/channel0']
    mask['B1'] = regions_h5[2]['t0/channel2']

    # The mining strategy used is highly empirical
    # Swap for other strategies or adjust values by trial and error

    # Create or open the dataset
    dataset = Dataset(dataset_path)
    print('preexisting keys : {}'.format(list(dataset.keys())))

    # Handle Regions one by one
    for region in regions[:1]:
        print('Processing Region '+region)
        # load image channel numpy array into working memory
        im = image[region][...]
        # load mask channel numpy array into working memory
        msk = mask[region][...]
        mean = np.mean(im)
        std = np.std(im)
        count, bins = np.histogram(im, bins=100, range=[0, 2000])
        # Save and print region statistics
        np.savetxt(output_directory + '/region_{}_histogramm_counts'.format(region),
                   count, delimiter=',', )  # save histogramm data to csv
        np.savetxt(output_directory + '/region_{}_histogramm_bins'.format(region),
                   bins, delimiter=',')  # save histogramm data to csv
        print(region)
        print('mean: {} std: {}'.format(mean, std))

        # Visualize the image channel histogram
        plt.figure()
        plt.hist(bins[:-1], bins, weights=count, log=True)
        plt.title('Image Channel Histogram for region ' + region)
        plt.ylabel('log(counts)')
        plt.ylabel('Signal intensity')
        plt.savefig(output_directory + '/region_'+region+'_hist.png')

        # Apply pre-processing to the image and mask arrays (copy in working memory)
        im = preprocessImage(
            im, plot_file=args.output_directory + '/region_' + region + '_expFit.png')
        msk = preprocessMask(msk)

        #########################################################
        # Mining training examples from the current region
        # Since most of the 3D Volume is empty space, a sampling strategy has to be used to specificaly target examples of interest.

        # Use a Unet Tiler to divide the region into a grid of training examples. They are enummerated by the tiler class and can be referenced by their index.
        #
        tiler = UnetTiler3D.forEntireCongruentData(image=im,
                                                   mask=msk,
                                                   output_shape=(132, 132, 132),
                                                   input_shape=(220, 220, 220))
        # get a list of random indices from the region
        # sample random tiles from the image volume
        indices = getRandomIndices(tiler, n_samples=200)

        # Calculate the fraction of foreground / object voxels in the training volumes
        mask_proportions = sampleMaskProportion(tiler, indices)

        # Prepare a detaset with mask thresholded sampling
        # thresholded sampling samples indices above the threshold with a higher probability
        # n_samples
        mask_thresholded = thresholdedSampling(indices, mask_proportions,
                                               threshold=0.001,
                                               above_threshold_ratio=args.above_threshold_ratio,
                                               n_samples=args.samples_per_region)

        # Add the selected training examples to the dataset
        dataset.add_tiles(tiler, mask_thresholded,
                          key_prefix=region,
                          # DO NOT CROP MASK TO OUPUT SIZE (if data augumentation is used, this prevents artifacts from affine and elastic deformations)
                          cropMask=False,
                          metadata={'region': region})

    # Clean up
    print('Dataset creation complete. Containes {} examples'.format(len(dataset)))

    # Close dataset
    dataset.close


############################################################################
#                              DATA PREPROCESSING
############################################################################
# The following function will be applied globaly PER REGION to normalize the image data
# Neural Networks rely on normalized image data (eg pixel values roughly in [-1,1] and comparable between specimens)
# WARNING Once a network has been trained on input that has been preprocessed by a certain preprocessing function, all the input to the network has to be preprocess in the same way !!!
def preprocessImage(x, plot_file=None):
    sf = calculateScalingFactor(x, plot_file=plot_file)
    x = scaleImage(x, sf)
    return x


# The following functions will be applied globaly PER REGION to preprocess the mask
def preprocessMask(x):
    # binarize mask and one hot encode
    x = np.clip(x, 0, 1)
    x = x.astype(np.int32)
    return x


if __name__ == "__main__":
    main()
