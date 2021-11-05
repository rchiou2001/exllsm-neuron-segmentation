"""
This script trains a Unet architecture and saves the trained model to a folder
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from elasticdeform import deform_random_grid
from tools.Dataset3D import Dataset
from tools.metrics import MeanIoU
from tools.utilities import (tf_occlude)
from unet.model import build_unet


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
        description='Training model for neuron segmentation')

    parser.add_argument('--training-dataset',
                        dest='training_dataset_path',
                        type=str, required=True,
                        help='Training dataset path')
    parser.add_argument('--output-dir',
                        dest='output_path',
                        type=str, required=True,
                        help='Output path')
    parser.add_argument('--model-name',
                        dest='model_file_name',
                        type=str,
                        default='resumed',
                        help='Model file name')
    parser.add_argument('--set_gpu_mem_growth', dest='set_gpu_mem_growth',
                        action='store_true', default=False,
                        help='If true set gpu memory growth')
    parser.add_argument('--resume',
                        dest='resume_training',
                        action='store_true',
                        default=False,
                        help='Resume training from an existing model. ' +
                             'Ensure that Architecture Parameters are ' +
                             'the same as in the pretrained model. ' +
                             'Training parameters may change.')
    parser.add_argument('--pre-trained-model',
                        dest='pre_trained_model_path',
                        type=str,
                        help='Pre-trained model path')

    # architecture parameters
    parser.add_argument('--initial-filters',
                        dest='initial_filters',
                        type=int,
                        default=1,
                        help='The number of filter maps in the first ' +
                             'convolution operation')
    parser.add_argument('--bottleneck-dropout-rate',
                        dest='bottleneck_dropout_rate',
                        type=float,
                        default=0.2,
                        help='Bottleneck dropout rate')
    parser.add_argument('--spatial-dropout',
                        dest='spatial_dropout',
                        action='store_true',
                        default=False,
                        help='Spatial drop out')
    parser.add_argument('--spatial-dropout-rate',
                        dest='spatial_dropout_rate',
                        type=float,
                        default=0.2,
                        help='Spatial dropout rate')

    # ATTENTION these parameters are not freely changable -> CNN arithmetics
    parser.add_argument('--n-blocks',
                        dest='n_blocks',
                        type=int,
                        default=2,
                        help='The number of Unet downsample/upsample blocks')
    parser.add_argument('--model_input_size',
                        dest='model_input_size', type=str,
                        metavar='dx,dy,dz', default='220,220,220',
                        help='Model input shape')
    parser.add_argument('--model_output_size',
                        dest='model_output_size', type=str,
                        metavar='dx,dy,dz', default='132,132,132',
                        help='Model output size')
    parser.add_argument('--library_size',
                        dest='library_size', type=str,
                        metavar='dx,dy,dz', default='220,220,220',
                        help='Size of examples in the training library')

    # Training parameters
    parser.add_argument('--test-fraction',
                        dest='test_fraction',
                        type=float,
                        default=0.2,
                        help='Fraction of training examples set aside for validation')
    parser.add_argument('--no-affine',
                        dest='no_affine',
                        action='store_true',
                        default=False,
                        help='No affine transform')
    parser.add_argument('--with-elastic-deform',
                        dest='elastic_deformation',
                        action='store_true',
                        default=False,
                        help='With elastic deformation')
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
    parser.add_argument('--epochs',
                        dest='n_epochs',
                        type=int,
                        default=3,
                        help='number of epochs to train the model')
    parser.add_argument('--object-class-weight',
                        dest='object_class_weight',
                        type=int,
                        default=5,
                        help='factor by which pixels showing the neuron ' +
                             'are multiplied in the loss function')
    parser.add_argument('--dice-weight',
                        dest='dice_weight',
                        type=float,
                        default=0.3,
                        help='contribution of dice loss (rest is cce)')
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        type=int,
                        default=1,
                        help='number of epochs to train the model')
    parser.add_argument('--with-visualization-sample',
                        dest='output_visualization_sample',
                        action='store_true',
                        default=False,
                        help='With occlusions')

    args = parser.parse_args()

    if args.set_gpu_mem_growth:
        _gpu_fix()

    # Ensure that output folder for diagnostics is created
    os.makedirs(args.output_path, exist_ok=True)

    model_input_size = tuple([int(d)
                              for d in args.model_input_size.split(',')])
    model_output_size = tuple([int(d)
                               for d in args.model_output_size.split(',')])
    library_size = tuple([int(d) for d in args.library_size.split(',')])

    affine_transform = False if args.no_affine else True

    def preprocess(x, y):
        # The unet expects the input data to have an additional channel axis.
        x = tf.expand_dims(x, axis=-1)
        y = tf.one_hot(y, depth=2, dtype=tf.int32)  # one hot encode to int tensor
        return x, y

    def crop_mask(x, y, mask_size=model_output_size):
        # apply crop after batch dimension is added x and y have (b,x,y,z,c) format while mask size has (x,y,z) format => add offset of 1
        crop = [(y.shape[d+1]-mask_size[d])//2 for d in range(3)]
        # keras implicitly assumes channels last format
        y = tf.keras.layers.Cropping3D(cropping=crop)(y)
        return x, y

    def occlude(x, y):
        return tf_occlude(x, y, occlusion_size=args.occlusion_size)

    def random_elastic_deform(x, y):
        # create a 5x5x5 grid of random displacement vectors
        # use order 0 (nearest neigbour interpolation for mask)
        x, y = elasticdeform.deform_random_grid([x, y],
                                                sigma=4,
                                                points=(5, 5, 5),
                                                order=[2, 0],
                                                mode="reflect",
                                                prefilter=False)
        return x, y

    def tf_random_elastic_deform(image: tf.Tensor, mask: tf.Tensor):
        image_shape = image.shape
        mask_shape = mask.shape
        image, mask = tf.numpy_function(random_elastic_deform,
                                        inp=[image, mask],
                                        Tout=(tf.float32, tf.int32))
        image.set_shape(image_shape)
        mask.set_shape(mask_shape)
        return image, mask

    print('Loading training dataset')
    # The Dataset3D class handles all file level i/o operations
    dataset = Dataset(args.training_dataset_path, append=False, readonly=True)
    # get a list of all records in the database and shuffle entries
    entries = list(dataset.keys())
    np.random.shuffle(entries)

    # Make a train test split and retrieve a callable ->
    # that produces a generator -> that yields the recods specified by the key list
    # in random order
    n_val = np.ceil(test_fraction*len(entries)).astype(np.int)
    training = dataset.getGenerator(entries[:-n_val])
    test = dataset.getGenerator(entries[-n_val:])

    # Instantiate tf Datasets from the generator producing callables, specify the datatype and shape of the generator output
    trainingset_raw = tf.data.Dataset.from_generator(training,
                                                     output_types=(tf.float32,
                                                                   tf.int32),
                                                     output_shapes=(tf.TensorShape(library_size),
                                                                    tf.TensorShape(library_size)))

    testset_raw = tf.data.Dataset.from_generator(test,
                                             output_types=(tf.float32,
                                                           tf.int32),
                                             output_shapes=(tf.TensorShape(library_size),
                                                            tf.TensorShape(library_size)))

    # the dataset is expected to be preprocessed (image normalized, mask binarized)

    # chain dataset transformations to construct the input pipeline for training

    # apply elastic deformations to raw dataset before expanding dimensions
    if args.elastic_deformation:
        #trainingset = trainingset.map(utilities.tf_elastic)
        # set the number of parallel calls to a value suitable for your machine (probably the number of logical processors)
        trainingset = trainingset_raw.map(
            tf_random_elastic_deform)  # , num_parallel_calls=9)
    else:
        trainingset = trainingset_raw  # just feed raw dataset into subsequent steps

    # expand dimensions of image and masl
    trainingset = trainingset.map(preprocess)
    # apply affine transformations
    if affine_transform:
        trainingset = trainingset.map(utilities.tf_affine)

    # apply occlusions
    if args.occlusions:
        trainingset = trainingset.map(occlude)

    trainingset = trainingset.batch(batch_size).map(crop_mask).prefetch(5)
    testset = testset_raw.map(preprocess).batch(
        batch_size).map(crop_mask).prefetch(5)

    # %% Construct model
    unet = build_unet(input_shape=model_input_size + (1,),
                      n_blocks=args.n_blocks,
                      initial_filters=args.initial_filters,
                      bottleneckDropoutRate=args.bottleneck_dropout_rate,
                      spatialDropout=args.spatial_dropout,
                      spatialDropoutRate=args.spatial_dropout_rate)

    # If we want to resume training load the pretrained model file instead
    if resume_training:
        print("Resuming training from model file at " + args.pre_trained_model_path)
        unet = tf.keras.models.load_model(args.pre_trained_model_path,
                                          custom_objects={
                                            "InputBlock" : model.InputBlock,
                                            "DownsampleBlock" : model.DownsampleBlock,
                                            "BottleneckBlock" : model.BottleneckBlock,
                                            "UpsampleBlock" : model.UpsampleBlock,
                                            "OutputBlock" : model.OutputBlock
                                          },
                                          compile=False)

    # Setup Training
    unet.compile(optimizer=tf.keras.optimizers.Adam(),
                 loss=model.weighted_cce_dice_loss(class_weights=[1,object_class_weight],
                                                   dice_weight=dice_weight),
                 metrics=['acc', MeanIoU(num_classes=2, name='meanIoU')]
    )
    # Train
    saved_model = args.output_path + '/' + args.model_file_name + '{epoch}.h5'
    log_filepath = args.output_path + '/' + args.model_file_name + '.log'
    history = unet.fit(trainingset,
                       epochs=args.n_epochs,
                       validation_data= testset,
                       verbose=1,
                       callbacks=[tf.keras.callbacks.ModelCheckpoint(saved_model),
                                  tf.keras.callbacks.CSVLogger(filename=log_filepath)
                                 ]
                      )
    # Evaluate

    # Generate some Plots from training history 
    # Plot the evolution of the training loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training','validation'])
    plt.title('Evolution of training loss')
    plt.xlabel('epochs')
    plt.ylabel('Spare Categorial Crossentropy')
    plt.savefig(args.output_path +'/loss.png')

    # Plot the evolution of pixel wise prediction accuracy
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Evolution of Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('categorial accuracy')
    plt.legend(['training', 'validation'])
    plt.savefig(args.output_path+'/accuracy.png')

    # Plot evolution of mean IoU Metric
    plt.figure()
    plt.plot(history.history['meanIoU'])
    plt.plot(history.history['val_meanIoU'])
    plt.title('Evolution of Mean IoU')
    plt.xlabel('epoch')
    plt.ylabel('mean intersection over union')
    plt.legend(['training', 'validation'])
    plt.savefig(args.output_path+'/iou.png')

    # %% Save some image mask pairs for visual inspection
    if (args.output_visualization_sample):
        import itertools
        import imageio
        tds = iter(trainingset)
        tds_raw = iter(trainingset_raw)
        for i in range(3):
            x, y = next(tds)
            xr, yr = next(tds_raw)
            imageio.volsave(args.output_path+"/image"+str(i)+".tif", x.numpy()[0, ..., 0])
            # extract foreground map and pad to original size
            y = y.numpy()[0, ..., 1]
            imageio.volsave(args.output_path+"/mask"+str(i)+".tif", np.pad(y, (44, 44)))
            imageio.volsave(args.output_path+"/mask_raw"+str(i)+".tif", yr)

        # Generate test image and apply deformation step manualy
        ti, tm = utilities.getTestImage(mask_size=(220, 220, 220), addAxis=False)
        tid, tmd = random_elastic_deform(ti, tm)
        imageio.volsave(args.output_path+"/testimage.tif", ti)
        imageio.volsave(args.output_path+"/testmask.tif", tm)
        imageio.volsave(args.output_path+"/testimage_deformed.tif", tid)
        imageio.volsave(args.output_path+"/testmask_deformed.tif", tmd)


if __name__ == "__main__":
    main()
