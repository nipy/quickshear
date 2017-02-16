#!/usr/bin/python
import numpy as np
import nibabel as nb
import sys
import logging


def edge_mask(mask):
    """Create an edge of brain mask from a binary brain mask.

    Return a two-dimensional edge of brain mask.
    """
    # Sagittal profile
    brain = mask.any(axis=0)

    # Simple edge detection
    edgemask = 4 * brain - np.roll(brain, 1, 0) - np.roll(brain, -1, 0) - \
                           np.roll(brain, 1, 1) - np.roll(brain, -1, 1) != 0
    return edgemask.astype('uint8')


def convex_hull(brain):
    """Use Andrew's monotone chain algorithm to find the lower half of the
    convex hull.

    Return a two-dimensional convex hull.
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
    for axis in np.nonzero(flips)[0]:
        data = nb.orientations.flip_axis(data, axis)
    return data


def orient_xPS(img, hemi='R'):
    """Set image orientation to RPS (or LPS), tracking flips for re-flipping"""
    axes = nb.orientations.aff2axcodes(img.affine)
    data = img.get_data()
    flips = np.array(axes) != np.array((hemi, 'P', 'S'))
    return flip_axes(data, flips), flips


def quickshear(anat_img, mask_img, buff=10):
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


def deface(anat_filename, mask_filename, defaced_filename, buff=10):
    """Deface neuroimage using a binary brain mask.

    Keyword arguments:
    anat_filename -- the filename of the neuroimage to deface
    mask_filename -- the filename of the binary brain mask
    defaced_filename -- the filename of the defaced output image
    buff -- the buffer size between the shearing line and the brain
        (default value is 10.0)
    """
    anat_img = nb.load(anat_filename)
    mask_img = nb.load(mask_filename)

    if anat_img.shape != mask_img.shape:
        logger.warning(
            "Anatomical and mask images do not have the same dimensions.")
        sys.exit(-1)

    new_anat = quickshear(anat_img, mask_img, buff)
    new_anat.to_filename(defaced_filename)
    logger.info("Defaced file: {0}".format(defaced_filename))


if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    # logging.basicConfig(filename="hull.log",level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if len(sys.argv) < 4:
        logger.debug(
            "Usage: quickshear.py anat_file strip_file defaced_file [buffer]")
        sys.exit(-1)
    else:
        anatfile = sys.argv[1]
        stripfile = sys.argv[2]
        newfile = sys.argv[3]
        if len(sys.argv) >= 5:
            try:
                buff = int(sys.argv[4])
            except:
                raise ValueError
            deface(anatfile, stripfile, newfile, buff)
        deface(anatfile, stripfile, newfile)
