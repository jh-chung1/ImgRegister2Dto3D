import os
import numpy as np
import scipy.ndimage
import matplotlib.image as img
from scipy.ndimage import rotate

def load_TS_tiff_img(img_dir):
    """
    Load a thin section image from a tiff file.
    Parameters:
        img_dir (str): Path to the image file.
    Returns:
        np.ndarray: Loaded and flipped TS image.    
    """
    TS_tiff_img = img.imread(os.path.join(img_dir))
    TS_tiff_img = np.transpose(TS_tiff_img, (1, 0, 2))
    TS_tiff_img = np.flip(TS_tiff_img.T, axis=0)
    return TS_tiff_img

def load_TS_img(img_dir, img_shape):
    """
    Load a thin section (TS) image from a binary file.
    
    Parameters:
        img_dir (str): Path to the image file.
        img_shape (tuple): Shape of the image (rows, columns).
    
    Returns:
        np.ndarray: Loaded and flipped TS image.
    """
    TS_np_img = np.fromfile(os.path.join(img_dir), np.uint8).reshape(img_shape[0], img_shape[1])
    TS_np_img = np.flip(TS_np_img.T, axis=0)
    return TS_np_img

def load_CT_img(img_dir, img_shape):
    """
    Load a CT image from a binary file.
    
    Parameters:
        img_dir (str): Path to the image file.
        img_shape (tuple): Shape of the image (depth, rows, columns).
    
    Returns:
        np.ndarray: Loaded CT image.
    """
    CT_img = np.fromfile(os.path.join(img_dir), np.uint8).reshape(img_shape[0], img_shape[1], img_shape[2])
    return CT_img

def resize_image(np_img, target_voxel_size, reference_voxel_size):
    """
    Resize an image to match the target voxel size with reference voxel size.
    
    Parameters:
        img (np.ndarray): Input image.
        target_voxel_size (float): Voxel size of the image.
        reference_voxel_size (float): Reference voxel size for resizing.
    
    Returns:
        np.ndarray: Resized image.
    """
    resizing_ratio = target_voxel_size / reference_voxel_size
    return scipy.ndimage.zoom(np_img, resizing_ratio)

def resize_for_testing(np_img, zoom_factor):
    """
    Further resize an image for testing purposes.
    
    Parameters:
        img (np.ndarray): Input image.
        zoom_factor (float): Factor by which to resize the image.
    
    Returns:
        np.ndarray: Resized image.
    """
    return scipy.ndimage.zoom(np_img, zoom_factor)

def rotate_3Dimg(image_3d, rotation_angles):
    """
    Rotates a 3D binary image around the x, y, and z axes.

    Parameters:
    - image_3d (numpy.ndarray): The input 3D binary array.
    - rotation_angles (list or tuple of three floats): Rotation angles (degrees) in the form [theta_x, theta_y, theta_z].

    Returns:
    - numpy.ndarray: The rotated 3D binary image.
    """
    theta_x, theta_y, theta_z = rotation_angles

    # Rotate along different axes
    rotated_x = rotate(image_3d, theta_x, axes=(1, 2), reshape=False, order=0)
    rotated_y = rotate(rotated_x, theta_y, axes=(0, 2), reshape=False, order=0)
    rotated_z = rotate(rotated_y, theta_z, axes=(0, 1), reshape=False, order=0)

    return rotated_z

def sect(rotated_3D, sect_no, section_axis=2):
    """
    Pick one section in rotated 3D image to specify "image" in template matching.
    section_axis: 0, 1, or 2. Default is 2.
    """
    if section_axis == 0:
        return rotated_3D[sect_no, :, :]
    elif section_axis == 1:
        return rotated_3D[:, sect_no, :]
    elif section_axis == 2:
        return rotated_3D[:, :, sect_no]
    else:
        raise ValueError("section_axis must be 0, 1, or 2")


def custom_initial_population(bounds, initial_values, popsize, dim):
    pop = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(popsize, dim)
    )
    pop[0, :] = initial_values
    pop[:, 3] = np.round(pop[:, 3])  # Ensure integer section index
    return pop

def load_real_Berea(CT_data_dir, Template_dir, zoom = 1.0):
    TS_img_shape = (8460, 13776)
    CT_img_shape = (1234, 748, 748)
    TS_voxel_size = 0.65e-6
    CT_voxel_size = 8.02e-6

    TS_img = load_TS_img(Template_dir, TS_img_shape)
    CT_img = load_CT_img(CT_data_dir, CT_img_shape)

    resized_TS_img = resize_image(TS_img, TS_voxel_size, CT_voxel_size)
    resized_TS_img = resize_for_testing(resized_TS_img, zoom)
    CT_img = resize_for_testing(CT_img, zoom)

    template = resized_TS_img

    angle_bounds = (0, 10)
    section_bounds = (CT_img.shape[1] // 4, CT_img.shape[1] // 4 * 3)
    bounds = [
        angle_bounds,
        (-3, 3),
        (-3, 3),
        section_bounds
    ]

    initial_values = np.array([5, 0, 0, round(130 * zoom)])
    population = custom_initial_population(bounds, initial_values, popsize=50, dim=4)
    mutation = (0.3, 1.2)

    return template, CT_img, bounds, population

def load_spherepack(CT_data_dir, Template_dir):
    CT_img_shape = (300, 300, 300)
    CT_img = load_CT_img(CT_data_dir, CT_img_shape)
    template = np.load(Template_dir)

    angle_bounds = (-7, 7)
    section_bounds = (CT_img.shape[0] // 5 * 2, CT_img.shape[0] // 5 * 3)
    bounds = [
        angle_bounds,
        angle_bounds,
        angle_bounds,
        section_bounds
    ]

    population = np.random.uniform(low=[-7, -5, -5, 120], high=[7, 5, 5, 180], size=(20, 4))
    population[0] = [0, 0, 0, 150]
    mutation = (0.3, 0.7)

    return template, CT_img, bounds, population