from .corr import normxcorr2
from .img_processing import rotate_3Dimg, sect
import numpy as np

def objective_function(params, image_3D, template, axis):
    """
    Objective function for Differential Evolution optimization.
    Finds the best rotation angles and section number to maximize the NCC peak.
    """
    theta_x, theta_y, theta_z, sect_no = params
    sect_no = int(round(sect_no))  # Ensure section number is an integer

    # Rotate the 3D volume
    rotated_3D = rotate_3Dimg(image_3D, [theta_x, theta_y, theta_z])

    # Extract the 2D section
    img = sect(rotated_3D, sect_no, axis)

    # Compute NCC
    _, nxcorr_peak, _, _ = normxcorr2(template, img)

    return -nxcorr_peak  # Minimize the negative correlation to maximize it

def get_record_loss_callback(CT_img, template, axis, loss_history):
    def record_loss(xk, convergence=None):
        loss = objective_function(xk, CT_img, template, axis)
        loss_history.append(loss)
        return False
    return record_loss

