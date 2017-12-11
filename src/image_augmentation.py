#!/usr/bin/env python

# import libraries
from scipy.ndimage.interpolation import rotate

def image_rotation(img, rotation_settings):
    """
    INPUT:
        img:
            the matrix to rotate, assumes 0, 1 indicies are YX
        rotation_settings:
            angle_max:
                max angles (if int, float, |angle_max|, else min and max)
            angle_ticks:
                int of incriment of ticks
    OUTPUT:
        rotates the images in the YX plane
    """
    # type validation
    if type(rotation_settings['ANGLE_MAX']) in VALID_NUMERIC_TYPES:
        angle_tuple = [-rotation_settings['ANGLE_MAX'], rotation_settings['ANGLE_MAX']]
    elif type(rotation_settings['ANGLE_MAX']) in [list, tuple]:
        angle_tuple = tuple(rotation_settings['ANGLE_MAX'])
    else:
        TypeError("key ANGLE_MAX for rotation_settings is not numeric or list like")

    # type validation
    if type(rotation_settings['ANGLE_TICKS']) in VALID_NUMERIC_TYPES:
        rotation_angle = np.random.choice(np.arange(
            angle_tuple[0],
            angle_tuple[1],
            rotation_settings['ANGLE_TICKS']
        ))
    else:
        raise TypeError("key ANGLE_TICKS is not int or float")

    return rotate(img, rotation_angle, reshape=False)

def crop_image(img, crop_factor):
    """
    INPUTS:
        img:
            imput numpy image
        factor:
            factor to crop (in Y then X dimention)
    OUTPUTS:
        cropped numpy image
    """
    # determine sizes
    y_size, x_size = img.shape

    # determine min bounds
    new_y = int((y_size//2) - (y_size//crop_factor[1]//2))
    new_x = int((x_size//2) - (x_size//crop_factor[0]//2))

    # determine half values
    half_y, half_x = [int(img.shape[i] // crop_factor[i]) for i in range(2)]

    return img[new_y: new_y + half_y, new_x: new_x + half_x]

def image_augment(img, dicom_info, settings):
    """
    INPUTS:
        img:
            the input image 2D matrix
        dicom_info:
            the dictionary to specify current dicom information
        settings:
            the settings dictionary
    OUTPUT:
        the augmented image 2D matrix
    """
    import pdb; pdb.set_trace()
    # test for ANGLE_MAX for rotation
    if hasattr(settings, "ANGLE_MAX") and hasattr(setting, "ANGLE_TICKS"):
        # rotate image
        img = image_rotation(img, settings["ROTATION_SETTINGS"])

    # test for CROP_FACTOR
    if hasattr(settings, "CROP_TARGET"):
        # determine resize factor
        resize_factor = np.array(dicom_info["fov"] / settings["CROP_TARGET"])

        # crop image
        img = crop_image(img, resize_factor)

    return img
