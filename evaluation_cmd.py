"""
This script evaluates the performance of a pretrained unet
"""

# %% Imports
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tools.Dataset3D as Dataset3D
import tools.metrics as metrics
import tools.visualization as visualization
import tools.utilities as utilities

import unet.model as model


def _gpu_fix():
    # Fix for tensorflow-gpu issues
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print('Physical GPUs:', len(gpus), 'Logical GPUs:', len(logical_gpus))


def main():
    parser = argparse.ArgumentParser(
        description='Eval model for neuron segmentation')

    parser.add_argument('--dataset',
                        dest='dataset_path',
                        type=str,
                        required=True,
                        help='Dataset path')
    parser.add_argument('--model-path',
                        dest='model_path',
                        type=str,
                        help='Model location')
    parser.add_argument('--output-dir',
                        dest='output_path',
                        type=str,
                        required=True,
                        help='Output path')
    parser.add_argument('--verbose',
                        dest='verbose',
                        action='store_true',
                        default=False,
                        help='Make it verbose')

    parser.add_argument('--model_shape',
                        dest='model_shape', type=str,
                        metavar='dx,dy,dz', default='132,132,132',
                        help='Model shape')

    # Training parameters
    parser.add_argument('--with-occlusions',
                        dest='occlusions',
                        action='store_true',
                        default=False,
                        help='With occlusions')
    parser.add_argument('--occlusion-size',
                        dest='occlusion_size',
                        type=int,
                        default=40,
                        help='Side length of occuled cubes in training examples')

    args = parser.parse_args()

    if args.set_gpu_mem_growth:
        _gpu_fix()

    # Ensure that output folder for diagnostics is created
    os.makedirs(args.output_path, exist_ok=True)

    # Number of examples to use for evaluation
    n_val = 40
    # List of entry keys defining the evaluation subset (overrides) n_val if specified.
    subset = None

    # visualize evaluation examples once every visualization_fraction instances
    visualization_intervall = 10

    # Output size of the unet. (Input size should be compatible with dataset)
    model_shape = tuple([int(d) for d in args.model_shape.split(',')])

    # small print helper function
    def printv(*pargs):
        if args.verbose:
            print(*pargs)

    eval_report = {}  # create dictonary for evaluation report

    # %% Load evaluation data

    # The Dataset3D class handles all file level i/o operations
    dataset = Dataset3D.Dataset(args.dataset_path, append=False, readonly=True)
    printv('Dataset {} contains {} records'.format(dataset, len(dataset)))
    printv('Dataset metadata entries:')
    printv(dataset.getAttributes().keys())

    if subset is None:
        # get a list of all records in the database
        entries = list(dataset.keys())
        # shuffle entries
        np.random.shuffle(entries)
        # Make a train test split and retrieve a callable -> that produces a generator -> that yields the recods specified by the key list in random order
        subset = entries[:n_val]
    else:
        # adjust number of validation examples to the provided list
        n_val = len(subset)

    printv('Using random subset of size {} :'.format(n_val) + str(subset))
    eval_report['evaluation dataset'] = args.dataset_path
    eval_report['model file'] = args.model_path
    eval_report['number of evaluation examples'] = n_val
    eval_report['evaluation example keys'] = subset
    # use the random subset generated above or reuse the keys of another run for comparability
    samples = dataset.getGenerator(subset, shuffle=False)

    # %% Build tf data input pipeline

    # Instantiate tf Datasets from the generator producing callables, specify the datatype and shape of the generator output
    validationset_raw = tf.data.Dataset.from_generator(samples,
                                                       output_types=(
                                                           tf.float32, tf.int32),
                                                       output_shapes=(tf.TensorShape([220, 220, 220]), tf.TensorShape([220, 220, 220])))
    # each entry is preprocessed by passing it through this function

    # EXPECT normalized image chanel !
    # EXPECT binarized masks
    def preprocess(x, y):
        # The unet expects the input data to have an additional channel axis.
        x = tf.expand_dims(x, axis=-1)
        # one hot encode to int tensor
        y = tf.one_hot(y, depth=2, dtype=tf.int32)
        return x, y

    # Crop
    def crop_mask(x, y, mask_size=model_shape):
        # apply crop after batch dimension is added x and y have (b,x,y,z,c) format while mask size has (x,y,z) format => add offset of 1
        crop = [(y.shape[d+1]-mask_size[d])//2 for d in range(3)]
        # keras implicitly assumes channels last format
        y = tf.keras.layers.Cropping3D(cropping=crop)(y)
        return x, y

    # Occlude parts of the input image
    def occlude(x, y):
        x, y = utilities.tf_occlude(x, y, occlusion_size=args.occlusion_size)
        return x, y

    validationset = validationset_raw.map(preprocess)

    # Add occlusions if specified by the user
    if args.occlusions:
        validationset = validationset.map(occlude)

    validationset = validationset.batch(1).map(crop_mask).prefetch(1)
    validationset_iter = iter(validationset)

    # Restore the trained model. Specify where keras can find custom objects that were used to build the unet
    unet = tf.keras.models.load_model(args.model_path, compile=False,
                                      custom_objects={"InputBlock": model.InputBlock,
                                                      "DownsampleBlock": model.DownsampleBlock,
                                                      "BottleneckBlock": model.BottleneckBlock,
                                                      "UpsampleBlock": model.UpsampleBlock,
                                                      "OutputBlock": model.OutputBlock})

    # Compile the model using dummy values for loss function (evaluation loss is not reported)
    unet.compile(loss=model.weighted_cce_dice_loss(class_weights=[1, 5], dice_weight=0.3),
                 metrics=['acc', metrics.MeanIoU(num_classes=2, name='meanIoU')])

    # %% Evaluate model

    # evaluate keras metrics
    loss, accuracy, iou = unet.evaluate(validationset, verbose=0)
    eval_report['pixel wise accuracy'] = accuracy
    eval_report['mean IoU'] = iou

    # set up lists to store ground truth and prediction values
    y_true, y_pred = [], []
    # set up list to hold precision, recall curves
    thresholds = np.linspace(0, 1, 21)
    precision_curves, recall_curves = [], []
    auc_scores = []
    # get an new iterator on the validation set
    validationset_iter = iter(validationset)
    for n, (im, msk) in enumerate(validationset_iter):
        # get prediction for image
        pred = unet.predict(im)
        # convert to pseudoprobability
        pred = tf.nn.softmax(pred, axis=-1)
        y_pred.append(pred.numpy()[0, ..., 1])
        # get binary y_true from mask
        y_true.append(msk.numpy()[0, ..., 1])

        # Calculate binary prediction performance on aggregated model output
        # Doing this every 20 steps limits the amount of RAM used to store y_true / y_pred
        if (n+1) % 20 == 0 or n+1 == n_val:
            y_true = np.stack(y_true, axis=0)
            y_pred = np.stack(y_pred, axis=0)
            batch_precision, batch_recall, batch_thresholds, batch_auc = metrics.precisionRecall(
                y_true, y_pred)
            # Extract values at relevant thresholds
            precision_curve, recall_curve = [], []
            for t in thresholds:
                index = np.sum(batch_thresholds < t)
                precision_curve.append(batch_precision[index])
                recall_curve.append(batch_recall[index])
            # Add curves and auc of batch to list
            precision_curves.append(precision_curve)
            recall_curves.append(recall_curve)
            auc_scores.append(batch_auc)
            # Clear y_true / y_pred
            y_true, y_pred = [], []

        # visualize some training examples and the unet ouput
        if (n+1) % visualization_intervall == 0:
            # set up save path
            savePath = args.output_path+'/val_{}'.format(n)
            # select tensor slices
            # crop image and convert to (x,y,z) format (im is z normalized)
            im = im.numpy()[0, 44:176, 44:176, 44:176, 0]
            # extract fist channel of mask for vis
            msk = msk.numpy()[0, ..., 1]
            pred = pred.numpy()[0, ..., 1]
            # create visualization
            visualization.showZSlices(im, channel=None, vmin=0, vmax=1,
                                      n_slices=6, title='Input Image', savePath=savePath+'_0_im.png')
            visualization.showZSlices(msk, channel=None, vmin=0, vmax=1,
                                      n_slices=6, title='True Mask', savePath=savePath+'_1_true.png')
            visualization.showZSlices(pred, channel=None, vmin=0, vmax=1, n_slices=6,
                                      title='Predicted Mask Pseudoprobability', savePath=savePath+'_2_pred.png')
            # Make an overlay of the true mask (red) and the predicted mask (green)
            mask_overlay = visualization.makeRGBComposite(
                r=msk[..., np.newaxis], g=pred[..., np.newaxis], b=None, gain=1.)
            visualization.showZSlices(mask_overlay, n_slices=6, mode='rgb',
                                      title='True Mask @red Prediction @green', savePath=savePath+'_3_overlay.png')
        printv('evaluated {}/{}'.format(n, n_val))  # give

    plt.figure()
    for recall, precision in zip(recall_curves, precision_curves):
        plt.plot(recall, precision)
    plt.title('Binary Classification Performance on batches')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    plt.savefig(args.output_path+'/PrecisionRecallBatch.png')

    # Calculate mean curves and mean auc
    recall_mean = np.mean(np.array(recall_curves), axis=0)
    precision_mean = np.mean(np.array(precision_curves), axis=0)
    auc_mean = np.mean(np.array(auc_scores))

    plt.figure()
    plt.plot(recall_mean, precision_mean)
    plt.title('Binary Classification Performance')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    plt.text(0.1, 0.1, 'roc auc : {}'.format(auc_mean))
    plt.savefig(args.output_path+'/PrecisionRecall.png')

    summary = '\nthreshold precision recall\n'
    for i, t in enumerate(thresholds):
        summary = summary + \
            '{:.2f} {:.2f} {:.2f}'.format(
                t, precision_mean[i], recall_mean[i]) + '\r\n'
    printv(summary)
    eval_report['binary classification performance'] = summary
    eval_report['mean auc'] = auc_mean

    # Write evaluation Report
    reportFile = open(args.output_path+'/report.txt', 'w')

    for key in eval_report.keys():
        reportFile.write(str(key)+' : ')
        reportFile.write(str(eval_report[key])+'\r\n')
    reportFile.close()


if __name__ == "__main__":
    main()
