#!/usr/bin/python
import sys
import argparse
import numpy as np
import nibabel as nb
import logging


def edge_mask(mask):
    """ Find the edges of a mask or masked image

    Parameters
    ----------
    mask : 3D array
        Binary mask (or masked image) with axis orientation LPS or RPS, and the
        non-brain region set to 0

    Returns
    -------
    2D array
        Outline of sagittal profile (PS orientation) of mask
    """
    # Sagittal profile
    brain = mask.any(axis=0)

    # Simple edge detection
    edgemask = 4 * brain - np.roll(brain, 1, 0) - np.roll(brain, -1, 0) - \
                           np.roll(brain, 1, 1) - np.roll(brain, -1, 1) != 0
    return edgemask.astype('uint8')


def convex_hull(brain):
    """ Find the lower half of the convex hull of non-zero points

    Implements Andrew's monotone chain algorithm [0].

    [0] https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain

    Parameters
    ----------
    brain : 2D array
        2D array in PS axis ordering

    Returns
    -------
    (2, N) array
        Sequence of points in the lower half of the convex hull of brain
    """
    # convert brain to a list of points in an n x 2 matrix where n_i = (x,y)
    pts = np.vstack(np.nonzero(brain)).T

    def cross(o, a, b):
        return np.cross(a - o, b - o)

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    return np.array(lower).T


def flip_axes(data, flips):
    """ Flip a data array along specified axes

    Parameters
    ----------
    data : 3D array
    flips : (3,) sequence of bools
        Sequence of indicators for whether to flip along each axis

    Returns
    -------
    3D array
    """
    for axis in np.nonzero(flips)[0]:
        data = nb.orientations.flip_axis(data, axis)
    return data


def orient_xPS(img, hemi='R'):
    """ Set image orientation to RPS or LPS

    Parameters
    ----------
    img : SpatialImage
        Nibabel image to be reoriented
    hemi : 'R' or 'L'
        Orientation of first axis of output image (default: 'R')

    Returns
    -------
    data : 3D array_like
        Re-oriented data array
    flips : (3,) sequence of bools
        Sequence of indicators of axes flipped
    """
    axes = nb.orientations.aff2axcodes(img.affine)
    data = img.get_data()
    flips = np.array(axes) != np.array((hemi, 'P', 'S'))
    return flip_axes(data, flips), flips


def quickshear(anat_img, mask_img, buff=10):
    """ Deface image using Quickshear algorithm

    Parameters
    ----------
    anat_img : SpatialImage
        Nibabel image of anatomical scan, to be defaced
    mask_img : SpatialImage
        Nibabel image of skull-stripped brain mask or masked anatomical
    buff : int
        Distance from mask to set shearing plane

    Returns
    -------
    SpatialImage
        Nibabel image of defaced anatomical scan
    """
    anat, anat_flip = orient_xPS(anat_img)
    mask, mask_flip = orient_xPS(mask_img)

    edgemask = edge_mask(mask)
    low = convex_hull(edgemask)
    xdiffs, ydiffs = np.diff(low)
    slope = ydiffs[0] / xdiffs[0]

    yint = low[1][0] - (low[0][0] * slope) - buff
    ys = np.arange(0, mask.shape[2]) * slope + yint
    defaced_mask = np.ones(mask.shape, dtype='bool')

    for x, y in zip(np.nonzero(ys > 0)[0], ys.astype(int)):
        defaced_mask[:, x, :y] = 0

    return anat_img.__class__(flip_axes(defaced_mask * anat, anat_flip),
                              anat_img.affine, anat_img.header.copy())


def main(cmd, *argv):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    parser = argparse.ArgumentParser(
        description='Quickshear defacing for neuroimages',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('anat_file', type=str,
                        help="filename of neuroimage to deface")
    parser.add_argument('mask_file', type=str,
                        help="filename of brain mask")
    parser.add_argument('defaced_file', type=str,
                        help="filename of defaced output image")
    parser.add_argument('buffer', type=float, nargs='?', default=10.0,
                        help="buffer size (in voxels) between shearing plane "
                        "and the brain")

    opts = parser.parse_args(argv)

    anat_img = nb.load(opts.anat_file)
    mask_img = nb.load(opts.mask_file)

    if anat_img.shape != mask_img.shape:
        logger.warning(
            "Anatomical and mask images do not have the same dimensions.")
        return -1

    new_anat = quickshear(anat_img, mask_img, opts.buffer)
    new_anat.to_filename(opts.defaced_file)
    logger.info("Defaced file: {0}".format(opts.defaced_file))


if __name__ == '__main__':
    sys.exit(main(*sys.argv))
