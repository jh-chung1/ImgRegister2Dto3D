import os
import argparse
import numpy as np
from utils.corr import normxcorr2, cross_correlation
from utils.img_processing import rotate_3Dimg, sect, load_TS_tiff_img, load_TS_img, load_CT_img, resize_image, resize_for_testing, load_real_Berea, load_spherepack
from utils.optim import objective_function, get_record_loss_callback
from utils.visualize import visualize_results
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import time
import json

## Input
parser = argparse.ArgumentParser()
parser.add_argument('--angle_range', type=float, default=5.0, help='rotation angle range')
parser.add_argument('--zoom', type=float, default=1.0, help='zoom')
parser.add_argument('--CT_data_dir', type=str, default='/scratch/users/jhchung1/github_test/ImgRegister2Dto3D.github/data/B1M1_CT_binary_8.02um_8bu_748x748x1234.raw', help='CT data dir')
parser.add_argument('--Template_dir', type=str, default='/scratch/users/jhchung1/github_test/ImgRegister2Dto3D.github/data/B1M1_TS_binary_10X_PPL_8bu_13776_8460.raw', help='Thin section segmented image dir')
parser.add_argument('--TS_tiff_data_dir', type=str, default='/scratch/users/jhchung1/github_test/ImgRegister2Dto3D.github/data/B1M1_TS_10X_PPL_13776_8460.tiff', help='Thin section tiff image dir')
parser.add_argument('--cpu_num', type=int, default=24, help='rotation number')
parser.add_argument('--test_type', type=str, choices=['Real_BereaSandstone', 'Validation_SpherePack'], default='Real_BereaSandstone', help='Type of test data')
parser.add_argument('--section_axis', type=int, choices=[0, 1, 2], default=2, help='Axis along which to extract 2D section')
args = parser.parse_args()

# *** Modify with your initial directory to save the results ***
optimization_result = f'/scratch/users/jhchung1/github_test/ImgRegister2Dto3D.github/results/{args.test_type}/optimization_results_zoom_{args.zoom}.json'
registered_img_dir = f'/scratch/users/jhchung1/github_test/ImgRegister2Dto3D.github/results/{args.test_type}/visualization_zoom_{args.zoom}.png'
os.makedirs(os.path.dirname(optimization_result), exist_ok=True)
os.makedirs(os.path.dirname(registered_img_dir), exist_ok=True)

# Specify test type
if args.test_type == 'Real_BereaSandstone':
    template, CT_img, bounds, initial_population = load_real_Berea(args.CT_data_dir, args.Template_dir, args.zoom)

elif args.test_type == 'Validation_SpherePack':
    template, CT_img, bounds, initial_population = load_spherepack(args.CT_data_dir, args.Template_dir)

else:
    raise NotImplementedError(f"Unsupported test_type: {args.test_type}")

if __name__ == "__main__":  
    start_time = time.time()  # Start timer (tic)
    print(f'Start optimiziation')
    
    # Record loss at each iteration
    loss_history = []
    callback = get_record_loss_callback(CT_img, template, args.section_axis, loss_history)

    # Run the optimization
    result = differential_evolution(
        objective_function, 
        bounds, 
        args=(CT_img, template, args.section_axis), 
        strategy='best1bin', 
        maxiter=2000,  
        popsize=50,  
        tol=1e-5,  
        mutation=(0.3, 1.2),  
        recombination=0.9,  
        workers=args.cpu_num,  
        updating="deferred",
        init=initial_population,
        callback=callback
    )

    end_time = time.time()  # End timer (toc)
    elapsed_time = end_time - start_time  # Compute elapsed time
    print(f"Optimization took {elapsed_time:.2f} seconds.")
    
    # Extract optimal parameters
    optimal_params = result.x
    optimal_correlation = -result.fun  # Convert back to positive
    optimal_section = int(round(optimal_params[3]))

    print("Optimal Rotation Angles (degrees):", optimal_params[:3])
    print("Optimal Section Number:", optimal_section)
    print("Maximum Correlation Coefficient:", optimal_correlation)

    # Visualize results
    (optimized_rotation_3D, optimized_section_in_3D, 
     nxcorr_map, nxcorr_peak, peak_row, peak_col) = visualize_results(
         CT_img, template, optimal_params, optimal_section
     )

    # Save optimization results
    results_dict = {
        "elapsed_time": elapsed_time,
        "optimal_params": optimal_params.tolist(),
        "optimal_section": optimal_section,
        "optimal_correlation": optimal_correlation,
        "nxcorr_peak": nxcorr_peak,
        "peak_row": int(peak_row),
        "peak_col": int(peak_col),
        "loss_history": loss_history
    } 
    with open(optimization_result, "w") as f:    
        json.dump(results_dict, f, indent=4)

    # Save visualization figure
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(optimized_section_in_3D, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(template, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(optimized_section_in_3D, cmap='gray')
    plt.gca().add_patch(
        plt.Rectangle((peak_col, peak_row), template.shape[1], template.shape[0], 
                      edgecolor='red', facecolor='none', label='NCC')
    )
    plt.legend()
    plt.savefig(registered_img_dir, dpi=200)
    plt.show()
