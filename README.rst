Quickshear
----------
Quickshear uses a skull stripped version of an anatomical images as a reference
to deface the unaltered anatomical image.

Usage::

    quickshear.py [-h] anat_file mask_file defaced_file [buffer]

    Quickshear defacing for neuroimages

    positional arguments:
      anat_file     filename of neuroimage to deface
      mask_file     filename of brain mask
      defaced_file  filename of defaced output image
      buffer        buffer size (in voxels) between shearing plane and the brain
                    (default: 10.0)

    optional arguments:
      -h, --help    show this help message and exit

For a full description, see the following paper:

.. [Schimke2011] Schimke, Nakeisha, and John Hale. "Quickshear defacing for neuroimages."
    Proceedings of the 2nd USENIX conference on Health security and privacy.
    USENIX Association, 2011.
