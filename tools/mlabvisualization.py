"""
Tools for visualization of 3D image data

Linus Meienberg
June 2020
"""
import numpy as np

from mayavi import mlab


# 3D Visualization tools
def show3DImage(image_tensor, channel=0, mode='image', newFigure=True, **kwargs):
    """
    Visualize a single channel 3D image using mayavi's isosurface plot

    Parameters
    ----------
    image_tensor : tensor
        tensor of rank 4 in format (x,y,z,c)
    channel : int, optional
        the channel to visualize, by default 0
    mode : str, optional
        the visualization mode. Set to {'image','mask'}
    """
    if mode is 'image':
        n_contours = 4
        transparent = True
    elif mode is 'mask':
        n_contours = 10
        transparent = False
    else:
        raise ValueError('Visualization mode undefined')
    if newFigure:
        mlab.figure(size=(500, 500))
    plot = mlab.contour3d(
        image_tensor[..., channel], contours=n_contours, transparent=transparent, **kwargs)
    return mlab.gcf()


def show3DDisplacementField(dx, dy, dz, mask_points=2000, scale_factor=50., newFigure=True, **kwargs):
    """Visualize a 3D vector field using mayavi.

    Parameters
    ----------
    dx, dy, dz : tensor
        rank three tensors holding the x,y,z components of the 3D vector field.
    mask_points : int, optional
        How many vectors to mask out for each vector displayed, by default 2000
    scale_factor : float, optional
        factor by which arrows are scaled for displaying the vector field, by default 50

    Returns
    -------
    mlab.figure
        Mayavi plot according to the running backend.
    """
    if newFigure:
        mlab.figure(size=(500, 500))
    plot = mlab.pipeline.vector_field(dx, dy, dz)
    mlab.pipeline.vectors(plot, mask_points=mask_points,
                          scale_factor=scale_factor, **kwargs)
    return mlab.gcf()


def showCutplanes(image_tensor, channel=0, hint=True, newFigure=True, **kwargs):
    """
    Shows an interactive plot with two movable cutplanes through a 3d volume

    Parameters
    ----------
    image_tensor : tensor
        tensor of rank 4 in format (x,y,z,c)
    channel : int, optional
        the channel to visualize, by default 0
    hint : bool, optional
        wheter to show a faint outline of the 3d volume, by default True
    newFigure : bool, optional
        wheter to create a new figure in mlab , by default True

    Returns
    -------
    [type]
        [description]
    """
    if newFigure:
        mlab.figure(size=(700, 700))
    s = image_tensor[..., channel]
    src = mlab.pipeline.scalar_field(s)
    mid_x = s.shape[0] // 2
    mid_y = s.shape[1] // 2
    if hint:
        mlab.pipeline.iso_surface(
            src, contours=[np.min(s)+0.8*s.ptp(), ], opacity=0.3, **kwargs)
    mlab.pipeline.image_plane_widget(src,
                                     plane_orientation='x_axes',
                                     slice_index=mid_x, **kwargs)
    mlab.pipeline.image_plane_widget(src,
                                     plane_orientation='y_axes',
                                     slice_index=mid_y, **kwargs)
    mlab.outline()
    return mlab.gcf()
