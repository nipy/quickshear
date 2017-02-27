#!/usr/bin/python
import numpy
import nibabel as nb
import sys
import logging


def edge_mask(mask):
    """Create an edge of brain mask from a binary brain mask.

    Return a two-dimensional edge of brain mask.
    """
    brain = numpy.zeros(mask.shape[1:])
    # iterate over axial
    for i in range(0, mask.shape[1] - 1):
        # iterate over coronal
        for k in range(mask.shape[2] - 1, 0, -1):
            brain[i, k] = mask[:, i, k].any()

    edgemask = numpy.zeros(brain.shape, dtype='uint8')
    for u in range(1, brain.shape[0] - 2):
        for v in range(1, brain.shape[1] - 2):
            if brain[u, v] + brain[u - 1, v] == 1:
                edgemask[u, v] = 1
            elif brain[u, v] + brain[u, v - 1] == 1:
                edgemask[u, v] = 1
            elif brain[u, v] + brain[u + 1, v] == 1:
                edgemask[u, v] = 1
            elif brain[u, v] + brain[u, v + 1] == 1:
                edgemask[u, v] = 1
    return edgemask


def convex_hull(brain):
    """Use Andrew's monotone chain algorithm to find the lower half of the
    convex hull.

    Return a two-dimensional convex hull.
    """
    # convert brain to a list of points
    nz = numpy.nonzero(brain)
    # transpose so we get an n x 2 matrix where n_i = (x,y)
    pts = numpy.array([nz[0], nz[1]]).transpose()

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for i in range(0, pts.shape[0]):
        p = (pts[i, 0], pts[i, 1])
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    return numpy.array(lower).transpose()


def deface(anat_filename, mask_filename, defaced_filename, buff=10):
    """Deface neuroimage using a binary brain mask.

    Keyword arguments:
    anat_filename -- the filename of the neuroimage to deface
    mask_filename -- the filename of the binary brain mask
    defaced_filename -- the filename of the defaced output image
    buff -- the buffer size between the shearing line and the brain
        (default value is 10.0)
    """
    nii_anat = nb.load(anat_filename)
    nii_mask = nb.load(mask_filename)

    if numpy.equal(nii_anat.shape, nii_mask.shape).all():
        pass
    else:
        logger.warning(
            "Anatomical and mask images do not have the same dimensions.")
        sys.exit(-1)

    anat_ax = nb.orientations.aff2axcodes(nii_anat.get_affine())
    mask_ax = nb.orientations.aff2axcodes(nii_mask.get_affine())

    logger.debug("Anat image axes: {0}".format(anat_ax))
    logger.debug("Mask image axes: {0}".format(mask_ax))
    logger.debug("Mask shape!: {0}".format(nii_mask.shape))

    mask = nii_mask.get_data()
    anat = nii_anat.get_data()

    anat_flip = [False, False, False]

    if anat_ax[0] != mask_ax[0]:
        # align mask to anat space
        logger.debug("Aligning mask to anatomical space... {0} -> {1}".format(
            mask_ax[0], anat_ax[0]))
        mask = nb.orientations.flip_axis(mask, 0)
    if anat_ax[1] != 'P':
        # flip anatspace
        logger.debug("Aligning anatomical image to +x -> P")
        anat_flip[1] = True
        anat = nb.orientations.flip_axis(anat, 1)
    if mask_ax[1] != 'P':
        # flip anatspace
        logger.debug("Aligning mask to +x -> P")
        mask = nb.orientations.flip_axis(mask, 1)
    if anat_ax[2] != 'S':
        # flip anatspace
        logger.debug("Aligning anatomical image to +y -> S")
        anat_flip[2] = True
        anat = nb.orientations.flip_axis(anat, 2)
    if mask_ax[2] != 'S':
        # flip anatspace
        logger.debug("Aligning mask to +y -> S")
        mask = nb.orientations.flip_axis(mask, 2)

    edgemask = edge_mask(mask)
    low = convex_hull(edgemask)
    slope = (low[1][0] - low[1][1]) / (low[0][0] - low[0][1])

    yint = low[1][0] - (low[0][0] * slope) - buff
    ys = numpy.arange(0, mask.shape[2]) * slope + yint
    defaced_mask = numpy.ones(mask.shape, dtype='uint8')

    for x in range(0, ys.size - 1):
        if ys[x] < 0:
            break
        else:
            ymax = min(ys[x], mask.shape[2])
            defaced_mask[:, x, :ymax] = 0

    defaced_img = defaced_mask * anat

    if anat_flip[1]:
        newimg = nb.orientations.flip_axis(defaced_img, 1)
    if anat_flip[2]:
        newimg = nb.orientations.flip_axis(defaced_img, 2)
    else:
        newimg = defaced_img
    new_anat = nb.Nifti1Image(newimg, nii_anat.affine, nii_anat.header.copy())
    nb.save(new_anat, defaced_filename)
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
