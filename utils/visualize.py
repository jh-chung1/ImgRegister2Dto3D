import matplotlib.pyplot as plt
from .img_processing import rotate_3Dimg
from .corr import normxcorr2

def visualize_results(image_3D, template, optimal_params, optimal_section):
    # save and visualize the results
    optimized_rotation_3D = rotate_3Dimg(image_3D, optimal_params[:3])   # rotate 3D image based on the optmized paramter set
    optimized_section_in_3D = optimized_rotation_3D[:, :, optimal_section] # extract 2D image
    nxcorr_map, nxcorr_peak, peak_row, peak_col = normxcorr2(template, optimized_section_in_3D) # perform normalize correlation coefficient calculation
        
    image = optimized_section_in_3D
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.subplot(1,3,2)
    plt.imshow(template)
    # Compare results with ground truth
    plt.subplot(1, 3, 3)
    plt.imshow(image, cmap='gray')
    # plt.gca().add_patch(
    #     plt.Rectangle((start_col, start_row), template.shape[1], template.shape[0], 
    #                   edgecolor='green', facecolor='none', label='Ground Truth')
    # )
    plt.gca().add_patch(
        plt.Rectangle((peak_col, peak_row), template.shape[1], template.shape[0], 
                      edgecolor='red', facecolor='none', label='NCC')
    )
    #plt.gca().add_patch(
    #    plt.Rectangle((pc_peak[1], pc_peak[0]), template.shape[1], template.shape[0], 
    #                  edgecolor='blue', facecolor='none', label='PC')
    #)
    plt.legend()
    plt.show()
    
    return optimized_rotation_3D, optimized_section_in_3D, nxcorr_map, nxcorr_peak, peak_row, peak_col 