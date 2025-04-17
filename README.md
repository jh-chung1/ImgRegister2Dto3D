# Image registration of 2D optical thin sections in a 3D porous medium: Application to a Berea sandstone digital rock image

![RegisteredImg](./readme_figs/Registered_TS_in_CT.png)
*(a) Registered thin section within the 3D CT volume of Berea sandstone, (b) Comparison between the registered image extracted from the Berea 3D volume and the segmented thin-section image*

## Abstract
This study proposes a systematic image registration approach to align 2D optical thin-section images within a 3D digital rock volume. Using template image matching with differential evolution optimization, we identify the most similar 2D plane in 3D. The method is validated on a synthetic porous medium, achieving exact registration, and applied to Berea sandstone, where it achieves a structural similarity index (SSIM) of 0.990. With the registered images, we explore upscaling properties based on paired multimodal images, focusing on pore characteristics and effective elastic moduli. The thin-section image reveals 50 \% more porosity and submicron pores than the registered CT plane. In addition, bulk and shear moduli from thin sections are 25 \% and 30 \% lower, respectively, than those derived from CT images. Beyond numerical comparisons, thin sections provide additional geological insights, including cementation, mineral phases, and weathering effects, which are not clear in CT images. This study demonstrates the potential of multimodal image registration to improve computed rock properties in digital rock physics by integrating complementary imaging modalities.

## Usage
### 1. Install the environment
```bash
conda env create -f environment.yml
conda activate ImgRegister2Dto3D
```
### 2. Run the script
```bash
python ImgReg_main.py --CT_data_dir <path_to_CT> --Template_dir <path_to_template> --test_type <Real_BereaSandstone or Validation_SpherePack> --cpu_num <num_cores>
```
#### Example for Berea sandstone (real)
```bash
python ImgReg_main.py \
--CT_data_dir data/B1M1_CT_binary_8.02um_8bu_748x748x1234.raw \
--Template_dir data/B1M1_TS_binary_10X_PPL_8bu_13776_8460.raw \
--test_type Real_BereaSandstone \
--cpu_num 24
```

#### Example for Sphere pack
```bash
python ImgReg_main.py \
--CT_data_dir data/SpherePack_300x300x300.raw \
--Template_dir data/SpherePack_template_Rotation_X_5_Y_4_Z_-3_SectNo_155.raw \
--test_type Validation_SpherePack \
--cpu_num 24
```

### 3. Input argument descriptions
```bash
--angle_range: Rotation angle range (default: 5.0)
--zoom: Zoom factor applied to the CT image (default: 1.0)
--CT_data_dir: Path to 3D CT volume (in .raw format)
--Template_dir: Path to binary thin section template (in .raw)
--TS_tiff_data_dir: Path to original thin section image (TIFF)
--cpu_num: Number of CPU cores for parallel optimization
--test_type: 'Real_BereaSandstone' or 'Validation_SpherePack'
--section_axis: Axis along which to extract 2D planes from CT (0, 1, or 2)
```


## Dataset
The dataset used in this study is publicly available on [Zenodo](https://zenodo.org/records/15237327)

## Citation
If you use this work or code in your research, please consider citing our [paper](https://arxiv.org/abs/2504.06604) 

    @article{chung2025image,
    title={Image registration of 2D optical thin sections in a 3D porous medium: Application to a Berea sandstone digital rock image},
    author={Chung, Jaehong and Cai, Wei and Mukerji, Tapan},
    journal={arXiv preprint arXiv:2504.06604},
    year={2025}
    }
