import os
import re
import shutil
import numpy as np
import pandas as pd
import nibabel as nib
from glob import glob
from nipype.interfaces import fsl as fsl
from nipype.interfaces.fsl import ApplyMask, ImageStats, FAST, DilateImage, UnaryMaths, BinaryMaths, Threshold, ErodeImage
from itertools import combinations
from nipype.interfaces.fsl.maths import MathsCommand
from traits.api import TraitError


# Helper function imports for directory and DICOM conversion operations
from utils import find_sub_dirs, convert_dicom_to_nii, remove_files_before_run

# Define paths for scripts, reference data, and subject directories for processing
# scripts_dir = '/mnt/z/Rotem_Orad/scripts/PhD'
# reference_dir = '/mnt/z/Shelly/'

subjects_dir_1 = '/mnt/z/Biogen_Shelly/biogen_first_time_point_shelly/'
subjects_dir_2 = '/mnt/z/Biogen_Shelly/biogen_second_time_point_shelly/'
subjects_dirs = [subjects_dir_1, subjects_dir_2]

subject_dir_two_timepoints = '/mnt/z/Biogen_Shelly/'
exclude_paths = ['nipype_bedpostx/', 'tbss/', 'tbss_non_FA/', 'stats/', 'FS_output/', 'Positive/', 'previous_results/', 'problematic/']
# change_to_LPI_files = ['star_map2.nii', "star_L_NC.nii.gz", "star_R_NC.nii.gz", "star_L_SN.nii.gz", "star_R_SN.nii.gz"]

# def change_to_LPI(directory):
#     for file in change_to_LPI_files:
#         img = nib.load(os.path.join(directory, file))
#         img_LPI = nib.as_closest_canonical(img)
#         # if file=='star_map2.nii':
#         #     file = file + '.gz'
#         nib.save(img_LPI, os.path.join(directory, file))

# def calc_vol_sn_t2star(star_image, nm_sn_seg):
    
    

def brain_extract(directory, filename, frac_threshold=0.4, vertical_gradient=0, padding=True):
    """
    Extracts the brain from an MRI image using FSL's BET tool with customizable options.
    """
    btr = fsl.BET()  # Initialize BET function from FSL.
    btr.inputs.in_file = os.path.join(directory, filename)  # Specify input file.
    btr.inputs.frac = frac_threshold  # Set fractional intensity threshold.
    btr.inputs.out_file = os.path.join(directory, f'brain_{filename}')  # Define output file.
    btr.inputs.vertical_gradient = vertical_gradient  # Set vertical gradient parameter.
    btr.inputs.padding = padding  # Enable/disable padding.
    btr.run()  # Execute the brain extraction.


def create_bilateral_mask(sub, left_mask, right_mask, output_filename="sliced_bi_sn_mask.nii.gz"):
    """
    Generates a bilateral mask by combining left and right hemisphere masks.
    """
    l_sn = np.asanyarray(nib.load(os.path.join(sub, left_mask)).dataobj)  # Load left hemisphere mask.
    r_sn = np.asanyarray(nib.load(os.path.join(sub, right_mask)).dataobj)  # Load right hemisphere mask.
    bi_sn = l_sn + r_sn  # Combine both masks to create a bilateral mask.
    ni_img = nib.Nifti1Image(bi_sn, affine=np.eye(4))  # Create a Nifti image from the combined mask.
    nib.save(ni_img, output_filename)  # Save the bilateral mask.

# Function to process T1-weighted MRI images, converting DICOM to NIfTI format as needed.
def process_t1(sub):
    """
    Processes T1-weighted MRI images by converting DICOM to NIfTI format if necessary.
    """
    # Iterate through specific T1-weighted image types: UNI, INV1, INV2.
    for mp_file in ['UNI', 'INV1', 'INV2']:
        # Check if NIfTI files for the T1 image type exist.
        if not glob(f"{sub}/*{mp_file}*.nii"):
            # Search subdirectories for matching DICOM files.
            for direct in find_sub_dirs(sub):
                # If a directory contains the T1 image type, convert DICOMs to NIfTI.
                if re.search(fr'({mp_file})+', direct):
                    convert_dicom_to_nii(direct)


# Run RemoveNoise_mp2rage.py

# Identify and convert neuromelanin-sensitive images to NIfTI format
def process_neuromelanin(sub):
    """
    Identifies and processes neuromelanin-sensitive MRI images, converting them as needed.
    """
    # Check if GRE-MT (neuromelanin-sensitive) NIfTI images exist.
    if not glob(f"{sub}/*gre_mt*.nii"):
        # Search subdirectories for GRE-MT DICOM files, avoiding specified paths.
        for direct in find_sub_dirs(sub, exclude_paths):
            # If a directory contains GRE-MT sequences, convert DICOMs to NIfTI.
            if re.findall(r'.+(gre_mt)+', direct):
                convert_dicom_to_nii(direct)

    # Process and organize NM-related NIfTI files, assigning them to specific echo times.
    if not glob(f"{sub}/TE24.nii"):
        for file in os.listdir(sub):
            # Assign files to echo times or NM based on naming conventions.
            if re.search(r'(e1)\.(nii)$', file):
                shutil.copyfile(os.path.join(sub, file), os.path.join(sub, 'TE8.nii'))
                os.rename(os.path.join(sub, file), os.path.join(sub, 'NM.nii'))
            elif re.search(r'(e2)\.(nii)$', file):
                os.rename(os.path.join(sub, file), os.path.join(sub, 'TE16.nii'))
            elif re.search(r'(e3)\.(nii)$', file):
                os.rename(os.path.join(sub, file), os.path.join(sub, 'TE24.nii'))


def bet_neuromelanin_manual_masks(sub):
    """
    Applies brain extraction to neuromelanin images using manual thresholds.
    """
    print("Applying BET on anatomical and NM images.")
    brain_extract(sub, "anat.nii.gz", vertical_gradient=0.5)  # For anatomical images.
    brain_extract(sub, "NM.nii", frac_threshold=0.5, vertical_gradient=0.5)  # For NM images.


def bbox_ND(img):
    """
    Calculates the bounding box for non-zero regions in an image, supporting multi-dimensional data.
    """
    N = img.ndim  # Get the number of dimensions in the image.
    out = []
    for ax in combinations(reversed(range(N)), N - 1):
        nonzero = np.any(img, axis=ax)  # Find non-zero regions along each axis.
        out.append(np.where(nonzero)[0][[0, -1]])  # Store the bounding indices.
    return tuple(out)  # Return bounding box as a tuple.
    

# Isolate specific brain regions based on provided masks for detailed analysis
def cut_da_image(sub, input, background_mask, mask_l, mask_r, out_cut='sliced_NM.nii.gz', contrast=''):
    """
    Extracts a region of interest from an image based on provided masks, useful for focusing on specific brain regions.
    """
    input_img = nib.load(os.path.join(sub, input)).get_fdata()
    seg_img = nib.load(os.path.join(sub, background_mask)).get_fdata()
    mask_l_img = nib.load(os.path.join(sub, mask_l)).get_fdata()
    mask_r_img = nib.load(os.path.join(sub, mask_r)).get_fdata()

    # Define bounding box dimensions and crop the image accordingly
    bbox_idxs = bbox_ND(seg_img)
    bbox_x, bbox_y, bbox_z = bbox_idxs
    bbox_factor = 320  # Factor to adjust bounding box size
    bbox_w = bbox_x[1] - bbox_x[0]
    bbox_h = bbox_y[1] - bbox_y[0]
    bbox_cx = bbox_x[1] - bbox_w / 2.
    bbox_cy = bbox_y[1] - bbox_h / 2.
    bbox_dim = max((bbox_w, bbox_h))
    bbox_x = (int(bbox_cx - bbox_dim * bbox_factor / 2), int(bbox_cx + bbox_dim * bbox_factor / 2))
    bbox_y = (int(bbox_cy - bbox_dim * bbox_factor / 2), int(bbox_cy + bbox_dim * bbox_factor / 2))
    bbox_slice = (slice(*bbox_x), slice(*bbox_y), slice(*bbox_z))
    img_out = input_img[bbox_slice]

    # Save cropped images and masks for further processing
    nib_out = nib.Nifti1Image(img_out.transpose([0, 1, 2]), affine=np.eye(4))
    nib.save(nib_out, os.path.join(sub, out_cut))
    background_mask_cut = seg_img[bbox_slice]
    nib_background_out = nib.Nifti1Image(background_mask_cut.transpose([0, 1, 2]), affine=np.eye(4))
    nib.save(nib_background_out, os.path.join(sub, f'sliced_{background_mask}'))
    
    mask_l_cut = mask_l_img[bbox_slice]
    nib_mask_out = nib.Nifti1Image(mask_l_cut.transpose([0, 1, 2]), affine=np.eye(4))
    nib.save(nib_mask_out, os.path.join(sub, f'sliced_{contrast}_{mask_l}'))
    
    mask_r_cut = mask_r_img[bbox_slice]
    nib_mask_out = nib.Nifti1Image(mask_r_cut.transpose([0, 1, 2]), affine=np.eye(4))
    nib.save(nib_mask_out, os.path.join(sub, f'sliced_{contrast}_{mask_r}'))


def process_subject_space(sub):
    """
    Processes images and masks within the subject's space, including slicing and mask generation.
    """
    cut_da_image(sub, "brain_NM.nii.gz", "MIDBRAIN.nii.gz", "L_SN.nii.gz", "R_SN.nii.gz")
    cut_da_image(sub, "brain_anat.nii.gz", "MIDBRAIN.nii.gz", "L_SN.nii.gz", "R_SN.nii.gz", "sliced_anat.nii.gz")
    try:
        create_bilateral_mask(sub, "star_L_SN.nii.gz", "star_R_SN.nii.gz", "star_bi_sn_mask.nii.gz")
        cut_da_image(sub, "star_map2.nii", "star_bi_sn_mask.nii.gz", "star_L_SN.nii.gz", "star_R_SN.nii.gz", "sliced_star_sn.nii.gz")
        create_bilateral_mask(sub, "star_L_NC.nii.gz", "star_R_NC.nii.gz", "star_bi_nc_mask.nii.gz")
        cut_da_image(sub, "star_map2.nii", "star_bi_nc_mask.nii.gz", "star_L_NC.nii.gz", "star_R_NC.nii.gz", "sliced_star_nc.nii.gz")
        cut_da_image(sub, "star_map2.nii", "MIDBRAIN.nii.gz", "L_SN.nii.gz", "R_SN.nii.gz", "sliced_star_nm_sn.nii.gz", "star_nm")
    except FileNotFoundError:
        print("T2star images not found")


def process_t2star_masks(sub, mask_list, contrast_input):
    """
    Applies masks to T2* star images and computes mean values within the masked regions.

    Parameters:
        sub (str): Subject directory path.
        mask_list (list): List of mask filenames.
        contrast_input (str): Input contrast image filename.
    """
    for mask in mask_list:
        ApplyMask(in_file=os.path.join(sub, contrast_input), out_file=os.path.join(sub, mask), mask_file=os.path.join(sub, mask)).run()
        mean = ImageStats(in_file=os.path.join(sub, mask), terminal_output='file', op_string='-M')
        mean.run()
        os.rename("output.nipype", os.path.join(os.path.join(sub, ''.join([mask, '_mask_mean.txt']))))


def neuromelanin_manual_masks_anat(sub):
    """
    Segments anatomical images using FSL's FAST tool and processes the resulting masks.
    """
    FAST(in_files=os.path.join(sub, "sliced_anat.nii.gz"), number_classes=3, output_type='NIFTI_GZ', segments=True).run()
    MathsCommand(in_file=os.path.join(sub, "sliced_anat_pve_2.nii.gz"), args='-thr %s -bin' % '0.99').run()


def process_val_background(sub):
    """
    Validates background noise levels and calculates SNR and CNR for neuromelanin images.
    Compute: 
    SNR = Mean_over_slices{(SigSN / SigBND) * 100)}
    CNR = Mean_over_slices{(SigSN -SigBND) / STDBND}
    SigSN is the signal intensity in SN ROI,
    SigBND the signal intensity in background ROI,
    STDBND the standard deviation in background ROI.
    """
    create_bilateral_mask(sub, "sliced__L_SN.nii.gz", "sliced__R_SN.nii.gz", "sliced_bi_sn_mask.nii.gz")
    if not os.path.isfile(os.path.join(sub, "sliced_background_mask.nii.gz")):
        print("Creating background mask by subtracting SN from the midbrain")
        midbrain = np.asanyarray(nib.load(os.path.join(sub, "sliced_midbrain.nii.gz")).dataobj)
        DilateImage(in_file="sliced_bi_sn_mask.nii.gz", operation='max', nan2zeros=True).run()
        bilateral_sn = np.asanyarray(nib.load(os.path.join(sub, "sliced_bi_sn_mask_dil.nii.gz")).dataobj)
        background = midbrain - bilateral_sn
        ni_img = nib.Nifti1Image(background, affine=np.eye(4))
        nib.save(ni_img, "sliced_background_mask.nii.gz")

    sub_snr = []
    sub_cnr = []

    # Apply background mask and calculate background statistics.
    ApplyMask(in_file=os.path.join(sub, "sliced_NM.nii.gz"), out_file=os.path.join(sub, 'nm_background.nii.gz'), mask_file=os.path.join(sub, "sliced_background_mask.nii.gz")).run()
    bg_mean = ImageStats(in_file=os.path.join(sub, "nm_background.nii.gz"), op_string='-M').run().outputs
    bg_mean = float(str(bg_mean).split()[-1])
    bg_std = ImageStats(in_file=os.path.join(sub, "nm_background.nii.gz"), op_string='-S').run().outputs
    bg_std = float(str(bg_std).split()[-1])
    print(f"Background mean: {bg_mean}, std: {bg_std}")

    # Process each specified mask to calculate SNR and CNR.
    for mask in ["sliced_bi_sn_mask", "sliced__L_SN", "sliced__R_SN"]:
        # Apply mask to image and calculate mean signal within the masked area.
        ApplyMask(in_file=os.path.join(sub, "sliced_NM.nii.gz"), out_file=os.path.join(sub, f'nm_{mask}.nii.gz'), mask_file=os.path.join(sub, f'{mask}.nii.gz')).run()
        mean_mask = ImageStats(in_file=os.path.join(sub, f'nm_{mask}.nii.gz'), op_string='-M').run().outputs
        mean_mask = float(str(mean_mask).split()[-1])
        print(f"Mean signal in {mask}: {mean_mask}")

        # Calculate SNR and CNR, store in lists, and write to files.
        snr = (mean_mask / bg_mean) * 100
        cnr = (mean_mask - bg_mean) / bg_std
        print(f"SNR for {mask}: {snr}, CNR for {mask}: {cnr}")
        sub_snr.append(snr)
        sub_cnr.append(cnr)
        with open(os.path.join(sub, f'bg_snr_{mask}_mask.txt'), "w+") as snr_file, open(os.path.join(sub, f'bg_cnr_{mask}_mask.txt'), "w+") as cnr_file:
            snr_file.write(str(snr))
            cnr_file.write(str(cnr))

        # Binarize the mask and calculate the mean signal within.
        UnaryMaths(in_file=os.path.join(sub, f'nm_{mask}.nii.gz'), operation='bin', out_file=os.path.join(sub, f'nm_{mask}_mask.nii.gz')).run()
        ImageStats(in_file=os.path.join(sub, "sliced_NM.nii.gz"), mask_file=os.path.join(sub, f'nm_{mask}_mask.nii.gz'), op_string='-M').run().outputs
        #os.rename("output.nipype", os.path.join(sub, f'manual_mean_{mask}_mask.txt'))

        # Calculate volume within the binarized mask.
        volume_mask = os.path.join(sub, f'nm_{mask}_mask.nii.gz')
        img = nib.load(volume_mask)
        nii_img = img.get_fdata()
        voxel_volume = np.prod(img.header.get_zooms())
        volume = np.sum(nii_img) * voxel_volume
        print(f"Volume within {mask}: {volume}")
        with open(os.path.join(sub, f'manual_vol_{mask}_mask.txt'), "w+") as vol_file:
            vol_file.write(str(volume))


def process_val_nawm(sub):
    """
    Validates and processes Normal-Appearing White Matter (NAWM) regions in brain images.
    """
    # Binarize and erode the white matter mask to isolate NAWM regions.
    MathsCommand(in_file=os.path.join(sub, "sliced_anat_pve_2_maths.nii.gz"), args='-sub 0.99 -bin').run()
    ErodeImage(in_file=os.path.join(sub, "sliced_anat_pve_2_maths_maths.nii.gz"), out_file="WM_ero.nii.gz").run()

    # Apply the eroded mask to the neuromelanin-sensitive image to segment NAWM.
    ApplyMask(in_file=os.path.join(sub, "sliced_NM.nii.gz"), out_file="NAWM.nii.gz", mask_file="WM_ero.nii.gz").run()
    # TODO delete midbrain 
    
    # Calculate the mean signal intensity within NAWM.
    mean_nawm = ImageStats(in_file=os.path.join(sub, "NAWM.nii.gz"), op_string='-M').run().outputs
    mean_nawm = float(str(mean_nawm).split()[-1])

    # Normalize neuromelanin-sensitive images based on the NAWM mean intensity.
    BinaryMaths(in_file=os.path.join(sub, "sliced_NM.nii.gz"), operand_value=mean_nawm, operation='div', out_file=os.path.join(sub, 'NM_normalized.nii.gz')).run()

    # Apply a threshold to identify regions with signal intensity above 1.1 times the mean NAWM intensity.
    Threshold(in_file=os.path.join(sub, 'NM_normalized.nii.gz'), thresh=1.1, out_file=os.path.join(sub, 'NM_thresholded.nii.gz')).run()

    # Process each specified mask to calculate statistics within thresholded regions.
    for mask in ["sliced_bi_sn_mask", "sliced__L_SN", "sliced__R_SN"]:
        # Apply the mask to the thresholded image and binarize the result.
        ApplyMask(in_file=os.path.join(sub, 'NM_thresholded.nii.gz'), out_file=os.path.join(sub, f'thresh_{mask}.nii.gz'), mask_file=os.path.join(sub, f'{mask}.nii.gz')).run()
        UnaryMaths(in_file=os.path.join(sub, f'thresh_{mask}.nii.gz'), operation='bin', out_file=os.path.join(sub, f'thresh_{mask}_mask.nii.gz')).run()

        # Calculate the mean signal intensity within the binarized, thresholded regions.
        mean = ImageStats(in_file=os.path.join(sub, 'NM_thresholded.nii.gz'), mask_file=os.path.join(sub, f'thresh_{mask}_mask.nii.gz'), op_string='-k %s -M').run().outputs
        mean = float(str(mean).split()[-1])
        with open(os.path.join(os.path.join(sub, ''.join(['thresh_mean_', mask, '.txt']))), "w+") as file:
            file.write(str(mean))

        # Calculate the volume of the binarized, thresholded regions.
        img = nib.load(os.path.join(sub, f'thresh_{mask}_mask.nii.gz'))
        voxel_count = np.sum(img.get_fdata() > 0)
        voxel_volume = np.prod(img.header.get_zooms())
        total_volume = voxel_count * voxel_volume
        print(f"Volume within {mask}: {total_volume}")
        with open(os.path.join(sub, f'volume_{mask}_mask.txt'), "w+") as vol_file:
            vol_file.write(str(total_volume))


def process_stat(subjects_dir): 
    """
    Compiles statistical data from processed images into an Excel spreadsheet for analysis.
    """
    # Initialize DataFrame with placeholder columns.
    df = pd.DataFrame(columns=["PatNum", "xxx"])
    df.set_index("PatNum")

    # Iterate through all text files containing statistical data.
    for elem in glob(os.path.join(subjects_dir, "*/*.txt")):
        elem_pat = elem.split("/")[-2]  # Extract subject identifier.
        elem_name = os.path.basename(elem).split("_mask")[0]  # Extract statistical measure name.
        elem_value = open(elem).read().strip()  # Read value from file.
        df.loc[elem_pat, elem_name] = elem_value  # Assign value to DataFrame.

    # Remove placeholder columns and save compiled data to Excel.
    df = df.drop(["PatNum", "xxx"], axis=1)
    df.to_excel(os.path.join(subjects_dir, "full_parse.xlsx"))  # Save compiled statistics.


def process_stat_two_timepoints(direct):
    """
    Aggregates statistical data from two timepoints into a single Excel file for longitudinal analysis.
    """
    arr = []  # Initialize list to hold data from each timepoint.

    # Collect Excel files from both timepoints.
    for elem in glob(os.path.join(direct, "*", "full_parse.xlsx")):
        df_in = pd.read_excel(elem)  # Load statistical data.
        df_in['Pat'] = elem.split("/")[-2]  # Add a column for subject identifier.
        arr.append(df_in)  # Append DataFrame to list.

    # Concatenate all DataFrames into a single DataFrame.
    df_all = pd.concat(arr)
    
    # Save the aggregated data to a new Excel file.
    df_all.to_excel(os.path.join(direct, "all_all.xlsx"))  # Save merged statistics from both timepoints.


def main():
    """
    Main function to orchestrate the processing of MRI images, including directory setup, image processing, and statistical analysis.
    """

    for subjects_dir in subjects_dirs:
    # Perform operations with each directory
        print(f"Processing directory: {subjects_dir}")

        for i, subject in enumerate(find_sub_dirs(subjects_dir, exclude_paths)):
            print(i)
            if i == 20:
                break
            
            subject_name = os.path.basename(subject.rstrip('/'))
            remove_files_before_run(subject)
            print(subject_name)
            # Copying T2star images
            try:
                if not os.path.exists(os.path.join(subjects_dir_2,subject, "star_map2.nii")):
                        original = os.path.join('//mnt/z/Rotem_Orad/Bio_second_timepoint', subject_name, "T2",  "star_map2.nii")
                        target = os.path.join(subjects_dir_2, subject, "star_map2.nii")
                        shutil.copyfile(original, target)
            except FileNotFoundError:
                print(f'{subject_name} missing t2star')
            # Change to subject directory and process images
            os.chdir(subject)
            # change_to_LPI(subject)
            process_t1(subject)
            process_neuromelanin(subject)
            bet_neuromelanin_manual_masks(subject)
            process_subject_space(subject)
            neuromelanin_manual_masks_anat(subject)
            process_val_background(subject)
            process_val_nawm(subject)

            # Error handling for T2star images
            try:
                process_t2star_masks(subject, ["sliced__star_L_SN.nii.gz", "sliced__star_R_SN.nii.gz"], "sliced_star_SN.nii.gz")
                process_t2star_masks(subject, ["sliced__star_L_NC.nii.gz", "sliced__star_R_NC.nii.gz"], "sliced_star_NC.nii.gz")
                process_t2star_masks(subject, ["sliced_star_nm_L_SN.nii.gz", "sliced_star_nm_R_SN.nii.gz"], "sliced_star_nm_sn.nii.gz")
            except TraitError:
                print("T2star images not found, pass")

    # Compile and process statistical data
        process_stat(subjects_dir)
    process_stat_two_timepoints(subject_dir_two_timepoints)
    

if __name__ == '__main__':
    main()