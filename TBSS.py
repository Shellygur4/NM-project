import fnmatch
import glob
import os
import re
import shutil
import nipype.interfaces.fsl as fsl
from basic_functions import find_sub_dirs, convert_dicom_to_nii, perform_brain_extraction
from edited_tbss_workflow import create_tbss_all, create_tbss_non_FA

# Define directory paths
scripts_dir = '//mnt/z/Rotem_Orad/scripts'
subjects_direct = '//mnt/z/Rotem_Orad/DLB_P_H/non_apoe'
tbss_direct = os.path.join(scripts_dir, 'TBSS')
fa_direct = os.path.join(tbss_direct, 'tbss')
non_fa_direct = os.path.join(tbss_direct, 'tbss_non_FA')
stat_dir = os.path.join(subjects_direct, 'stats')
path_avoid = [
    'output', 'stats/', 'FS_output/',
    'previous_results/', 'fsaverage/', 'results_may_23/', 'new_subs/'
]

# Constants
ALL_PERMISSIONS = 0o777
IMAGE_TYPES = ['FA', 'MD', 'Dr', 'Da']

def preprocess_dti(subjects_dir):
    for sub in find_sub_dirs(subjects_dir, path_avoid):
        subname = os.path.basename(sub)
        os.chdir(sub)
        print(subname)

        # Check for MP2RAGE files and convert them if needed
        mp2rage_files = glob.glob("mp2rage_denoised.nii")
        if not mp2rage_files:
            for file in find_sub_dirs(sub):
                # Check if the file matches the MP2RAGE naming pattern
                if re.search(r'(Se[0-1][0-9]).+(mp2rage)+', file):
                    nii_filename = convert_dicom_to_nii(file)
                    mp2rage_files.append(nii_filename)
                    # Run RemoveNoise

        # Check for DTI files and convert them if needed
        dti_files = glob.glob("*DTI*.nii")
        if not dti_files:
            for file in find_sub_dirs(sub):
                # Check if the file matches the DTI naming pattern
                if re.search(r'DFC(_MIX)?/$', file):
                    nii_filename = convert_dicom_to_nii(file)
                    dti_files.append(nii_filename)

        # Create the 'output' directory if it doesn't exist
        output_directory = os.path.join(sub, 'output')
        os.makedirs(output_directory, exist_ok=True)

        # Define the files you want to copy
        files_to_copy = {
            '*DTI*.nii': 'DTI4D.nii',
            '*DTI*.bvec': 'bvecs.bvec',
            '*DTI*.bval': 'bvals.bval',
        }

        # Copy the specified files if they don't exist in the 'output' directory
        for pattern, output_name in files_to_copy.items():
            matching_files = fnmatch.filter(os.listdir(sub), pattern)
            for file in matching_files:
                source_path = os.path.join(sub, file)
                destination_path = os.path.join(output_directory, output_name)

                if not os.path.exists(destination_path):
                    shutil.copyfile(source_path, destination_path)

        os.chdir(output_directory)
        if not os.path.exists('brain_DTI4D.nii.gz'):
            # Perform brain extraction on the corrected DWI data
            perform_brain_extraction(output_directory, 'DTI4D.nii')
        if not os.path.exists('eddy_corrected.nii.gz'):
            # Perform Eddy Current Correction
            eddy_correct(output_directory)

        # Fit diffusion tensor model
        DTIFit(output_directory)

        # Calculate Da
        Da = fsl.ApplyMask(
            in_file=os.path.join(output_directory, 'DTI__L1.nii.gz'),
            out_file=os.path.join(output_directory, 'Da.nii.gz'),
            mask_file=os.path.join(output_directory, 'brain_DTI4D_mask.nii.gz')
        )
        Da.run()

        # Calculate Dr
        tmpDr = fsl.BinaryMaths(
            in_file=os.path.join(output_directory, 'DTI__L2.nii.gz'),
            operand_file=os.path.join(output_directory, 'DTI__L3.nii.gz'),
            operation='add',
            out_file=os.path.join(output_directory, 'tmpDr.nii.gz')
        )
        tmpDr.run()

        Dr = fsl.BinaryMaths(
            in_file=os.path.join(output_directory, 'tmpDr.nii.gz'),
            operand_value=2.0,
            operation='div',
            out_file=os.path.join(output_directory, 'Dr.nii.gz')
        )
        Dr.run()

# Eddy Correction
def eddy_correct(direct):
    eddycorrect = fsl.EddyCorrect()
    eddycorrect.inputs.ref_num = 0
    eddycorrect.inputs.in_file = os.path.join(direct, 'brain_DTI4D.nii.gz')
    eddycorrect.inputs.out_file = os.path.join(direct, 'eddy_corrected.nii.gz')
    eddycorrect.run()

def DTIFit(direct):
    # Fit a diffusion tensor model at each voxel
    os.chdir(direct)
    dti = fsl.DTIFit()
    dti.inputs.dwi = os.path.join(direct, 'eddy_corrected.nii.gz')
    dti.inputs.bvecs = os.path.join(direct, 'bvecs.bvec')
    dti.inputs.bvals = os.path.join(direct, 'bvals.bval')
    dti.inputs.base_name = 'DTI_'
    dti.inputs.mask = os.path.join(direct, 'brain_DTI4D_mask.nii.gz')
    dti.run()

# TBSS functions
def tbss_FA(fa_list, base_dir):
    # Perform TBSS on FA images
    tbss_wf = create_tbss_all(base_dir=base_dir, estimate_skeleton=True)
    tbss_wf.inputs.inputnode.skeleton_thresh = 0.2
    tbss_wf.inputs.inputnode.fa_list = fa_list
    tbss_wf.run()

def tbss_non_FA(parm_list, field_list, base_dir):
    # Perform TBSS on non-FA images
    tbss_no_fa = create_tbss_non_FA(base_dir=base_dir)
    tbss_no_fa.inputs.inputnode.file_list = parm_list
    tbss_no_fa.inputs.inputnode.field_list = field_list
    tbss_no_fa.inputs.inputnode.skeleton_thresh = 0.2
    tbss_no_fa.inputs.inputnode.groupmask = os.path.join(tbss_direct, 'tbss/tbss3', 'groupmask',
                                                         'DTI__FA_prep_warp_merged_mask.nii.gz')
    tbss_no_fa.inputs.inputnode.meanfa_file = os.path.join(tbss_direct, 'tbss/tbss3', 'meanfa',
                                                           'DTI__FA_prep_warp_merged_masked_mean.nii.gz')
    tbss_no_fa.inputs.inputnode.all_FA_file = os.path.join(tbss_direct,'tbss/tbss3', 'mergefa',
                                                           'DTI__FA_prep_warp_merged.nii.gz')
    tbss_no_fa.inputs.inputnode.distance_map = os.path.join(tbss_direct, 'tbss/tbss4', 'distancemap',
                                                            'DTI__FA_prep_warp_merged_mask_inv_dstmap.nii.gz')
    tbss_no_fa.run()

def TBSS(subjects_dir):
    ALL_PERMISSIONS = 0o777
    fa_list = []
    md_list = []
    da_list = []
    dr_list = []
    field_list = []
    
    os.chdir(subjects_dir)
    if not os.path.exists(tbss_direct):
        os.makedirs(tbss_direct, mode=ALL_PERMISSIONS)

    for sub in find_sub_dirs(subjects_dir, path_avoid):
        directory = os.path.join(sub, 'output')
        fa_list.append(os.path.join(directory, 'DTI__FA.nii.gz'))
        md_list.append(os.path.join(directory, 'DTI__MD.nii.gz'))
        da_list.append(os.path.join(directory, 'Da.nii.gz'))
        dr_list.append(os.path.join(directory, 'Dr.nii.gz'))
    
    os.chdir(tbss_direct)
    tbss_FA(fa_list, tbss_direct)

    map_dir = os.path.join(fa_direct, 'tbss2/fnirt/mapflow')

    for subdir in find_sub_dirs(map_dir):
        field_list.append(os.path.join(subdir, 'DTI__FA_prep_fieldwarp.nii.gz'))


    if not os.path.exists(non_fa_direct):
        os.makedirs(non_fa_direct, mode=ALL_PERMISSIONS)
    os.chdir(non_fa_direct)
    
    for parm_list, folder_name, nii_pattern in [(md_list, 'MD', '*/*MD*.nii.gz'), (da_list, 'Da', '*/*Da*.nii.gz'), (dr_list, 'Dr', '*/*Dr*.nii.gz')]:
        print(parm_list)
        tbss_non_FA(parm_list, field_list, tbss_direct)
        target_dir = os.path.join(non_fa_direct, folder_name)
        
        if not os.path.exists(target_dir):
            os.mkdir(target_dir, ALL_PERMISSIONS)
        
        for path in glob.glob(nii_pattern):
            shutil.copyfile(path, os.path.join(target_dir, os.path.basename(path)))
            print('copied')

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, mode=0o777)

def run_randomise(in_file, mask, tcon, design_mat):
    randomise = fsl.Randomise(
        in_file=in_file,
        mask=mask,
        tcon=tcon,
        design_mat=design_mat,
        tfce2D=True
    )
    randomise.run()

def stats(tbss_dir, non_fa_dir, stat_dir, subjects_direct):
    """
    Feeds the 4D projected FA data into GLM modeling and thresholding in order to find voxels which correlate
    with the model.
    """
    create_directory(stat_dir)

    type_file = ['FA', 'MD', 'Dr', 'Da']
    files = [
        os.path.join(fa_direct, 'tbss4/projectfa/DTI__FA_prep_warp_merged_masked_skeletonised.nii.gz'),
        os.path.join(non_fa_dir, 'MD/DTI__MD_warp_merged_masked_skeletonised.nii.gz'),
        os.path.join(non_fa_dir, 'Dr/Dr_warp_merged_masked_skeletonised.nii.gz'),
        os.path.join(non_fa_dir, 'Da/Da_warp_merged_masked_skeletonised.nii.gz')
    ]

    for ftype in type_file:
        create_directory(os.path.join(stat_dir, ftype))

    os.chdir(tbss_direct)
    count = 0
    for file in files:
        run_randomise(in_file=file,
            mask=os.path.join(fa_direct, 'tbss4/skeletonmask/DTI__FA_prep_warp_merged_masked_mean_skeleton_mask.nii.gz'),
            tcon=os.path.join(subjects_direct, 'stat.con'),
            design_mat=os.path.join(subjects_direct, 'stat.mat'))
        
        randomized_count = 0
        for path in glob.glob('randomise*.nii.gz'):
            copied_path = shutil.copyfile(path, os.path.join(stat_dir, type_file[count], os.path.basename(path)))
            print(f"#{randomized_count} - copied file: {path} to: {copied_path}")
            randomized_count += 1

        count += 1
        print(f"produced {randomized_count} files")

def extract_values(direct):
    """Create an image of significant changes in TBSS."""
    diction = {'FA': 'FA/randomise_tfce_corrp_tstat1.nii.gz', 'MD': 'MD/randomise_tfce_corrp_tstat2.nii.gz',
               'Dr': 'Dr/randomise_tfce_corrp_tstat2.nii.gz', 'Da': 'Da/randomise_tfce_corrp_tstat2.nii.gz'}
    for key in diction.keys():
        math = fsl.ImageMaths(in_file=os.path.join(direct, diction[key]), op_string='-thr 0.95', out_file=os.path.join(
            direct, key, 'significant_results_mask.nii.gz'))
        math.run()

# Main function
def main():
    # preprocess_dti(subjects_direct)
    #TBSS(subjects_direct)
    stats(tbss_direct, non_fa_direct, stat_dir, subjects_direct)
    # extract_values(stat_dir)

if __name__ == '__main__':
    main()
