import fnmatch
import glob
import os
import re
import shutil
from basic_functions import find_sub_dirs, remove_files, convert_dicom_to_nii, perform_brain_extraction
import nibabel as nib
from nipype.interfaces import fsl
from nipype.interfaces.dcm2nii import Dcm2niix
from nipype.interfaces.freesurfer.preprocess import ReconAll
from nipype.interfaces.freesurfer.preprocess import MRIConvert
from nipype.interfaces.fsl import ImageStats
from nipype.interfaces.fsl.maths import BinaryMaths
from nipype.interfaces.fsl.maths import Threshold
from nipype.interfaces.fsl.maths import UnaryMaths
from nipype.interfaces.fsl.maths import BinaryMaths
from nipype.interfaces.fsl.model import Cluster
from nipype.interfaces.fsl.preprocess import FLIRT

# from niflow.nipype1.workflows.dmri.fsl.tbss import create_tbss_all
from edited_tbss_workflow import create_tbss_all, create_tbss_non_FA

# Run on the os fsl and freesurfer are installed
# When finished run script remove_noise
scripts_dir = '//mnt/z/Rotem_Orad/scripts'
reference_dir = '//mnt/z'
subjects_direct = '//mnt/z/Rotem_Orad/DLB_P_H/new_subs'
tbss_direct = os.path.join(scripts_dir, 'tbss')
non_fa_direct = os.path.join(scripts_dir, 'tbss_non_FA')
stat_dir = os.path.join(subjects_direct, 'stats')
path_avoid = ['nipype_bedpostx/', 'tbss/', 'tbss_non_FA/','output', 'stats/', 'FS_output/', 'previous_results/', 'fsaverage/', 'results_INA/', 'previous_results_apoe_non_only/']
regions = {
    '1': 'Left_Cerebral_Exterior',
    '2': 'Left_Cerebral_White_Matter',
    '3': 'Left_Cerebral_Cortex',
    '4': 'Left_Lateral_Ventricle',
    '5': 'Left_Inf_Lat_Vent',
    '7': 'Left_Cerebellum_White_Matter',
    '8': 'Left_Cerebellum_Cortex',
    '9': 'Left_Thalamus',
    '10': 'Left_Thalamus_Proper',
    '11': 'Left_Caudate',
    '12': 'Left_Putamen',
    '13': 'Left_Pallidum',
    '14': 'Third_Ventricle',
    '15': 'Fourth_Ventricle',
    '16': 'Brain_Stem',
    '17': 'Left_Hippocampus',
    '18': 'Left_Amygdala',
    '24': 'CSF',
    '26': 'Left_Accumbens_area',
    '28': 'Left_VentralDC',
    '40': 'Right_Cerebral_Exterior',
    '41': 'Right_Cerebral_White_Matter',
    '42': 'Right_Cerebral_Cortex',
    '43': 'Right_Lateral_Ventricle',
    '44': 'Right_Inf_Lat_Vent',
    '46': 'Right_Cerebellum_White_Matter',
    '47': 'Right_Cerebellum_Cortex',
    '48': 'Right_Thalamus',
    '49': 'Right_Thalamus_Proper',
    '50': 'Right_Caudate',
    '51': 'Right_Putamen',
    '52': 'Right_Pallidum',
    '53': 'Right_Hippocampus',
    '54': 'Right_Amygdala',
    '58': 'Right_Accumbens_area',
    '60': 'Right_VentralDC',
    '72': 'Fifth_Ventricle',
    '75': 'Left_Lateral_Ventricles',
    '76': 'Right_Lateral_Ventricles',
    '77': 'WM_hypointensities',
    '78': 'Left_WM_hypointensities',
    '79': 'Right_WM_hypointensities',
    '80': 'non_WM_hypointensities',
    '81': 'Left_non_WM_hypointensities',
    '82': 'Right_non_WM_hypointensities',

}

def perform_brain_extraction(directory, filename, frac=0.5):
    """
    Performs brain extraction on the given filename within the provided directory, creating skull-stripped brain masks.
    """
    btr = fsl.BET(in_file=os.path.join(directory, filename),
                  frac=frac,
                  out_file=os.path.join(directory, 'nodif_brain.nii.gz'),
                  functional=True)
    btr.run()


def brain_extract(direct, filename, fract=0.4):
    """
    The method gets a filename and a directory and
    creates masks of the brain without the skull
    """
    # filename= os.path.join(directory, filename)
    btr = fsl.BET()
    btr.inputs.in_file = (os.path.join(direct, filename))
    btr.inputs.frac = fract
    btr.inputs.out_file = (os.path.join(direct, ''.join(['brain_', filename])))
    btr.inputs.functional = True
    btr.run()


def DTIFit(direct):
    """
    The method gets a direct and fits a diffusion tensor model at each voxel
    """
    os.chdir(direct)
    dti = fsl.DTIFit()
    dti.inputs.dwi = os.path.join(direct, 'nodif_brain.nii.gz')  # if eddy correct works 'nodif_brain_res.nii.gz'
    dti.inputs.bvecs = os.path.join(direct, 'bvecs.bvec')
    dti.inputs.bvals = os.path.join(direct, 'bvals.bval')
    dti.inputs.base_name = 'DTI_'
    dti.inputs.mask = os.path.join(direct, 'nodif_brain_mask.nii.gz')
    dti.run()


def DTI(subjects_dir):
    """
    The function gets a subjects directory, loops over subjects folders and extracts diffusion tensors
        Outputs: <basename>_V1 - 1st eigenvector
                <basename>_V2 - 2nd eigenvector
                <basename>_V3 - 3rd eigenvector
                <basename>_L1 - 1st eigenvalue
                <basename>_L2 - 2nd eigenvalue
                <basename>_L3 - 3rd eigenvalue
                <basename>_MD - mean diffusivity
                <basename>_FA - fractional anisotropy (isotropic ~ 0; stick-like ~1)
                <basename>_MO - mode of the anisotropy (oblate ~ -1; isotropic ~ 0; prolate ~ 1)
                <basename>_S0 - raw T2 signal with no diffusion weighting
        """
    # loop over subjects folder
    for sub in find_sub_dirs(subjects_dir, path_avoid):
        os.chdir(sub)
        remove_files(sub, ['JustName.mat', 'Analysis', 'Logs'])
        if not glob.glob("*mp2rage*.nii"):  # Checking there are no files matching the pattern
            for file in find_sub_dirs(sub):
                if re.search(r'(Se[0-1][0-9]).+(mp2rage)+', file):  # if 'mp2rage' in file without Gd:
                    convert_dicom_to_nii(file)  # All mp2rge files are converted so later we can rum remove noise script
                if not re.search(r'(Se[0-1][0-9]).+(mp2rage)+', file):  # if 'mp2rage' in file without Gd:
                    if re.search(r'(mprage)+', file):
                        convert_dicom_to_nii(file)
                        os.rename(os.path.join(file, ".nii"), os.path.join('mp2rage_denoised.nii.gz'))
        if not glob.glob("*DTI*.nii"):
            for file in find_sub_dirs(sub):
                # file.endswith('DFC/')
                if re.search(r'DFC(_MIX)?/$', file):
                    convert_dicom_to_nii(file)
        directory = os.path.join(sub, 'output')
        if not os.path.exists(directory):
            os.mkdir(os.path.join(sub, 'output'))
        if not os.listdir(directory):
            # saving all the relevant files under 'output' folder
            for file in os.listdir('.'):
                if os.path.isfile(os.path.join(sub, file)):
                    if re.search('DTI', file):
                        if fnmatch.fnmatch(file, '**.nii'):
                            shutil.copyfile(os.path.join(sub, file),
                                            os.path.join(sub, 'output', 'DTI4D.nii'))
                        elif fnmatch.fnmatch(file, '**.bvec'):
                            shutil.copyfile(os.path.join(sub, file),
                                            os.path.join(sub, 'output', 'bvecs.bvec'))
                        elif fnmatch.fnmatch(file, '**.bval'):
                            shutil.copyfile(os.path.join(sub, file),
                                            os.path.join(sub, 'output', 'bvals.bval'))
        os.chdir(directory)

        
        perform_brain_extraction(directory, 'DTI4D.nii')
        # fsl.epi.EddyCorrect(in_file=os.path.join(directory, 'nodif_brain.nii.gz'),
        #                    out_file=os.path.join(directory, 'nodif_brain_res.nii.gz'), ref_num=0).run()
        DTIFit(directory)
        '''calc Da'''

        Da = fsl.ApplyMask(in_file=os.path.join(directory, 'DTI__L1.nii.gz'),
                           out_file=os.path.join(directory, 'Da.nii.gz'),
                           mask_file=os.path.join(directory, 'nodif_brain_mask.nii.gz'))
        Da.run()
        '''calc Dr '''
        tmpDr = fsl.BinaryMaths(in_file=os.path.join(directory, 'DTI__L2.nii.gz'),
                                operand_file=os.path.join(directory, 'DTI__L3.nii.gz'), operation='add',
                                out_file=os.path.join(directory, 'tmpDr.nii.gz'))
        tmpDr.run()
        Dr = fsl.BinaryMaths(in_file=os.path.join(directory, 'tmpDr.nii.gz'), operand_value=2.0, operation='div',
                             out_file=os.path.join(directory, 'Dr.nii.gz'))
        Dr.run()


def tbss_FA(fa_list, base_direct):
    """
    The method gets a list of all FA files and the base dir and does preproc, nonlinear registration of all FA images
    into standard space, creates the mean FA image and skeletonise it  as well as projecting all subjects' FA data
    onto the mean FA skeleton
    """
    tbss_wf = create_tbss_all('tbss', estimate_skeleton=True)
    tbss_wf.inputs.inputnode.skeleton_thresh = 0.2
    tbss_wf.inputs.inputnode.fa_list = fa_list
    tbss_wf.inputs.inputnode.base_dir = base_direct
    tbss_wf.run()


def tbss_non_FA(parm_list, field_list, base_direct):
    """
    The method is similar to TBSS but does it on other diffusion-derived data than FA images
    """
    tbss_no_fa = create_tbss_non_FA()
    tbss_no_fa.inputs.inputnode.file_list = parm_list
    tbss_no_fa.inputs.inputnode.field_list = field_list
    tbss_no_fa.inputs.inputnode.skeleton_thresh = 0.2
    tbss_no_fa.inputs.inputnode.groupmask = os.path.join(base_direct, 'tbss3', 'groupmask',
                                                         'DTI__FA_prep_warp_merged_mask'
                                                         '.nii.gz')
    tbss_no_fa.inputs.inputnode.meanfa_file = os.path.join(base_direct, 'tbss3', 'meanfa',
                                                           'DTI__FA_prep_warp_merged_masked_mean.nii.gz')
    tbss_no_fa.inputs.inputnode.all_FA_file = os.path.join(base_direct, 'tbss3', 'mergefa',
                                                           'DTI__FA_prep_warp_merged.nii.gz')
    tbss_no_fa.inputs.inputnode.distance_map = os.path.join(base_direct, 'tbss4', 'distancemap',
                                                            'DTI__FA_prep_warp_merged_mask_inv_dstmap.nii.gz')
    tbss_no_fa.inputs.inputnode.base_dir = base_direct
    tbss_no_fa.run()


def TBSS(subjects_dir):
    """
    The method implements tbss and non-FA tbss on the data
    """
    ALL_PERMISSIONS = 0o777
    fa_list = []
    md_list = []
    da_list = []
    dr_list = []
    field_list = []
    os.chdir(subjects_dir)
    if not os.path.exists(tbss_direct):
        os.mkdir(tbss_direct, ALL_PERMISSIONS)
    for sub in find_sub_dirs(subjects_dir, path_avoid):
        directory = os.path.join(sub, 'output')
        fa_list.append(os.path.join(directory, 'DTI__FA.nii.gz'))
        md_list.append(os.path.join(directory, 'DTI__MD.nii.gz'))
        da_list.append(os.path.join(directory, 'Da.nii.gz'))
        dr_list.append(os.path.join(directory, 'Dr.nii.gz'))
    os.chdir(tbss_direct)
    tbss_FA(fa_list, tbss_direct)
    for subdir in find_sub_dirs(os.path.join(tbss_direct, 'tbss2/fnirt/mapflow')):
        field_list.append(os.path.join(subdir, 'DTI__FA_prep_fieldwarp.nii.gz'))
    if not os.path.exists(non_fa_direct):
        os.mkdir(non_fa_direct)
        os.chdir(non_fa_direct)
    tbss_non_FA(md_list, field_list, tbss_direct)
    if not os.path.exists(os.path.join(non_fa_direct, 'MD')):
        os.mkdir(os.path.join(non_fa_direct, 'MD'), ALL_PERMISSIONS)
    for path in glob.glob('*/*MD*.nii.gz'):
        shutil.copyfile(path, os.path.join(os.path.join(non_fa_direct, 'MD'), os.path.basename(path)))
    if not os.path.exists(os.path.join(non_fa_direct, 'Da')):
        os.mkdir(os.path.join(non_fa_direct, 'Da'), ALL_PERMISSIONS)
    tbss_non_FA(da_list, field_list, tbss_direct)
    for path in glob.glob('*/*Da*.nii.gz'):
        shutil.copyfile(path, os.path.join(os.path.join(non_fa_direct, 'Da'), os.path.basename(path)))
    if not os.path.exists(os.path.join(non_fa_direct, 'Dr')):
        os.mkdir(os.path.join(non_fa_direct, 'Dr'), ALL_PERMISSIONS)
    tbss_non_FA(dr_list, field_list, tbss_direct)
    for path in glob.glob('*/*Dr*.nii.gz'):
        shutil.copyfile(path, os.path.join(os.path.join(non_fa_direct, 'Dr'), os.path.basename(path)))


def recon_all_func(direct):
    # Run script remove noise before running recon
    # important! fsaverage folder with the labels should be located in the subjects_dir direct (in our case:
    # FS_output) important! the file 'mni151_2mm.nii.gz' should be located in the FS_output folder.

    """Uses recon - all to generate surfaces and parcellations of structural data from anatomical images of a
    subject. """
    recon_all = ReconAll()
    recon_all.inputs.directive = 'all'
    recon_all.inputs.subjects_dir = direct
    recon_all.inputs.T1_files = os.path.join(direct, 'mp2rage_denoised.nii')
    recon_all.run()


def recon(subjects_dir):
    # Write in terminal: "sudo chmod -R a+w $SUBJECTS_DIR"
    # change the current dir to the relevant dir
    if not os.path.exists(os.path.join(subjects_dir, 'FS_output')):
        shutil.copytree('//mnt/z/Rotem_Orad/FS_output', os.path.join(subjects_dir, 'FS_output'))
    for sub in find_sub_dirs(subjects_dir, path_avoid):
        print(sub)
        os.chdir(sub)
        recon_all_func(sub)


def stats(tbss_dir, non_fa_dir):
    # important! create the model in GLM and save as stat
    """Feeds the 4D projected FA data into GLM modelling and thresholding in order to find voxels which correlate
    with the model. """
    ALL_PERMISSIONS = 0o777

    type_file = ['FA', 'MD', 'Dr', 'Da']
    files = [os.path.join(tbss_dir, 'tbss4/projectfa/DTI__FA_prep_warp_merged_masked_skeletonised.nii.gz'),
             os.path.join(non_fa_dir, 'MD/DTI__MD_warp_merged_masked_skeletonised.nii.gz'),
             os.path.join(non_fa_dir, 'Dr/Dr_warp_merged_masked_skeletonised.nii.gz'),
             os.path.join(non_fa_dir, 'Da/Da_warp_merged_masked_skeletonised.nii.gz')]
    if not os.path.exists(stat_dir):
        os.mkdir(stat_dir, ALL_PERMISSIONS)
    for ftype in type_file:
        if not os.path.exists(os.path.join(stat_dir, ftype)):
            os.mkdir(os.path.join(stat_dir, ftype), ALL_PERMISSIONS)

    count = 0
    for file in files:
        fsl.Randomise(
            in_file=file,
            mask=os.path.join(tbss_dir, 'tbss4/skeletonmask/DTI__FA_prep_warp_merged_masked_mean_skeleton_mask'
                                        '.nii.gz'),
            tcon= os.path.join(subjects_direct, 'stat.con'), design_mat=os.path.join(subjects_direct, 'stat.mat'), tfce2D=True).run()
        randomized_count = 0
        for path in glob.glob('randomise*.nii.gz'):
            copied_path = shutil.copyfile(path, os.path.join(os.path.join(stat_dir, type_file[count]), os.path.basename(path)))
            print(f"#{randomized_count} - copied file: {path} to: {copied_path}")
            randomized_count += 1
            
        count += 1
        print(f"produced {randomized_count} files")


def extract_values(direct):
    """Uses the directory of the statistics to create an image of the significant changes in TBSS. """
    diction = {'FA': 'FA/randomise_tfce_corrp_tstat1.nii.gz', 'MD': 'MD/randomise_tfce_corrp_tstat2.nii.gz',
               'Dr': 'Dr/randomise_tfce_corrp_tstat2.nii.gz', 'Da': 'Da/randomise_tfce_corrp_tstat2.nii.gz'}
    for key in diction.keys():
        print(os.path.join(direct, diction[key]))
        math = fsl.ImageMaths(in_file=os.path.join(direct, diction[key]), op_string='-thr 0.9', out_file=os.path.join(
            direct, key, 'significant_results_mask.nii.gz'))
        math.run()


def dce_brain_extraction(direct, filename):
    """
    The method gets a filename and a direct and
    creates masks of the brain without the skull
    """
    # filename= os.path.join(direct, filename)
    btr = fsl.BET()
    btr.inputs.in_file = (os.path.join(direct, filename))
    btr.inputs.frac = 0.5
    btr.inputs.out_file = (os.path.join(direct, 'DCE4D_vol0002_brain.nii.gz'))
    btr.inputs.functional = True
    btr.run()


def create_dce_mask(direct, region_number, region_name):
    """
    The method gets a direct,  region_number and region_name and
    creates masks of the given region registrated to DCE space
    """
    region_mask = os.path.join(direct, 'recon_all', 'mri',
                               ''.join(['r2DCE_aseg.nii.gz']))
    region_mask_img = nib.load(region_mask)
    region_mask_data = region_mask_img.get_fdata()
    region_mask_data[region_mask_data != int(region_number)] = 0

    new_region_mask = nib.Nifti1Image(region_mask_data, region_mask_img.affine, region_mask_img.header)
    nib.save(new_region_mask, os.path.join(
        os.path.join(direct, 'recon_all', 'mri',
                     ''.join(['r2DCE_', region_name, '_mask.nii.gz']))))

    cl = Cluster()
    cl.inputs.in_file = os.path.join(os.path.join(direct, 'recon_all', 'mri',
                                                  ''.join(['r2DCE_', region_name, '_mask.nii.gz'])))
    cl.inputs.threshold = int(region_number)
    cl.inputs.out_size_file = os.path.join(os.path.join(direct, 'recon_all', 'mri',
                                                        ''.join(['r2DCE_', region_name, '_cl_mask'])))
    cl.run()

    thr = Threshold()
    thr.inputs.in_file = os.path.join(os.path.join(direct, 'recon_all', 'mri',
                                                   ''.join(['r2DCE_', region_name, '_cl_mask.nii.gz'])))
    thr.inputs.thresh = 10
    thr.inputs.out_file = os.path.join(os.path.join(direct, 'recon_all', 'mri',
                                                    ''.join(['r2DCE_', region_name, '_cl_mask.nii.gz'])))
    thr.run()

    # Convert to bin mask:
    binary = UnaryMaths()
    binary.inputs.in_file = os.path.join(
        os.path.join(direct, 'recon_all', 'mri', 'r2DCE_', region_name, '_cl_mask.nii.gz'))
    binary.inputs.operation = 'bin'
    binary.inputs.out_file = os.path.join(
        os.path.join(direct, 'recon_all', 'mri', 'r2DCE_', region_name, '_cl_mask.nii.gz'))
    binary.run()


def calc_Ktrans2_in_mask(direct, region_name):
    """
    The method gets a direct and region_name and a threshold and calculates Ktrans2N in the given region by the
    following steps: 1. Multiply Ktrans2N map by the given region binary mask (creates Ktrans2N_region_name.nii.gz
    file) 2. Convert to bin mask 3. Extract volume (in voxels and mm^3) in the whole region and in the area that been
    obtained in section 4. 4. Calculates the median value of Ktrans2N in the area that been obtained in section 4.
    """
    # Multiply Ktrans2N map by the given region binary mask (creates Ktrans2N_region_name.nii.gz file)
    ktrans2 = BinaryMaths()
    ktrans2.inputs.in_file = os.path.join(direct, 'DCE', 'Ktrans2N.nii')
    ktrans2.inputs.operation = 'mul'
    ktrans2.inputs.operand_file = os.path.join(direct, 'recon_all' 'mri', 'r2DCE_', region_name,
                                               '_cl_mask.nii.gz')
    ktrans2.inputs.nan2zeros = True
    ktrans2.inputs.out_file = os.path.join(direct, 'DCE', ''.join(['Ktrans2N_no_thr_', region_name, '.nii.gz']))
    ktrans2.run()

    # Convert to bin mask:
    binary = UnaryMaths()
    binary.inputs.in_file = os.path.join(direct, 'DCE', 'Ktrans2N_no_thr_', region_name, '.nii.gz')
    binary.inputs.operation = 'bin'
    binary.inputs.out_file = os.path.join(direct, 'DCE', 'Ktrans2N_', region_name, '_no_thr_bin_mask',
                                          '.nii.gz')
    binary.run()

    # Extract volume (in voxels and mm^3) in the whole region:
    all_volume = ImageStats()
    all_volume.inputs.in_file = os.path.join(direct, 'recon_all', 'mri', 'r2DCE_', region_name, '_cl_mask.nii'
                                                                                                '.gz')
    all_volume.inputs.op_string = '-V'
    all_volume.terminal_output = 'file'
    all_volume.run()
    os.rename("output.nipype",
              os.path.join(direct, 'DCE', 'Volumes_MP2RAGE_no_thr', region_name, '_cl_bin_mask', '.txt'))

    # Extract volume (in voxels and mm^3) in the area that been obtained in section 2
    partial_volume = ImageStats()
    partial_volume.inputs.in_file = os.path.join(direct, 'DCE', 'Ktrans2N_', region_name, '_no_thr_bin_mask',
                                                 '.nii.gz')
    partial_volume.inputs.op_string = '-V'
    partial_volume.terminal_output = 'file'
    partial_volume.run()
    os.rename("output.nipype", os.path.join(direct, 'DCE', 'Volumes_MP2RAGE_no_thr', 'Ktrans2N_', region_name,
                                            '_no_thr_bin_mask', '.txt'))

    # Calculate median Ktrans2N value:
    median = ImageStats()
    median.inputs.in_file = os.path.join(direct, 'DCE', 'Ktrans2N.nii')
    median.terminal_output = 'file'
    median.inputs.mask_file = os.path.join(direct, 'DCE', 'Ktrans2N_', region_name, '_no_thr_bin_mask', '.nii'
                                                                                                        '.gz')
    median.inputs.op_string = '-k %s -P 50 '
    median.run()
    os.rename("output.nipype",
              os.path.join(direct, 'DCE', direct, 'Ktrans2N_no_thr', 'median_DCE_Ktrans2N_no_thr_',
                           region_name, '.txt'))

    # Calculate mean Ktrans2N value:
    mean_Ktrans2N = fsl.ImageMeants()
    mean_Ktrans2N.inputs.in_file = os.path.join(direct, 'DCE', 'Ktrans2N.nii')
    mean_Ktrans2N.inputs.mask = os.path.join(direct, 'DCE', 'Ktrans2N_', region_name, '_no_thr_bin_mask',
                                             '.nii.gz')
    mean_Ktrans2N.inputs.out_file = os.path.join(direct, 'DCE', direct, 'Ktrans2N_no_thr',
                                                 'mean_DCE_Ktrans2N_no_thr_', region_name, '.txt')
    mean_Ktrans2N.run()


def dce_summarize_results(direct, region_name, dce_map):
    """
    The method gets region name and dce_map name and summarize all the results in the current direct to one united
    txt file direct is the direct of the DCE files (for example: lupus/DCE)
    """

    with open(os.path.join(direct, dce_map, '_mean_', region_name, '_no_thr_.txt'), 'w') as outfile:
        for sub in find_sub_dirs(direct, path_avoid):
            with open(os.path.join(sub, dce_map, 'mean_DCE_', dce_map, '_', region_name, '.txt')) as infile:
                outfile.write(infile.read())


def dce_summarize_volumes(direct, region_name):
    with open(os.path.join(direct, 'DCE', 'Volumes_MP2RAGE_no_thr', 'volume_', region_name, '_.txt'),
              'w') as outfile:
        for sub in find_sub_dirs(direct, path_avoid):
            with open(
                    os.path.join(sub, 'DCE', 'Volumes_MP2RAGE_no_thr', region_name, '_cl_bin_mask', '.txt')) as infile:
                outfile.write(infile.read())


def dce_summarize_sub_volumes(direct, region_name, dce_map):
    with open(os.path.join(direct, 'DCE', 'Volumes_MP2RAGE_no_thr', dce_map, '_volume_', region_name, '_.txt'),
              'w') as outfile:
        for sub in find_sub_dirs(direct, path_avoid):
            with open(
                    os.path.join(sub, 'DCE', 'Volumes_MP2RAGE_no_thr', 'Ktrans2N', '_', region_name, '_no_thr_bin_mask',
                                 '.txt')) as infile:
                outfile.write(infile.read())


def DCE(subjects_dir):
    # Make sure your DCE analysis is in a folder named DCE

    for sub in find_sub_dirs(subjects_dir):
        try:
            os.mkdir(os.path.join(sub, 'DCE', 'Volumes_MP2RAGE_no_thr'))
        except FileExistsError:
            pass
        try:
            os.mkdir(os.path.join(sub, 'DCE'))
        except FileExistsError:
            pass
        try:
            os.mkdir(os.path.join(sub, 'DCE', 'Ktrans2N_no_thr'))
        except FileExistsError:
            pass
        if not os.listdir(os.path.join(sub, 'DCE', 'Ktrans2N_no_thr')):

            # extracting the third volume from DCE4D
            split = fsl.Split(in_file=os.path.join(sub, 'DCE', 'DCE4D.nii'), dimension='t')
            split.run()

            # brain extraction to the third volume from DCE4D
            dce_brain_extraction(os.path.join(sub, 'DCE'), 'DCE4D_vol0002.nii.gz')

            # converting brain.mgz and aseg.mgz to nii files:
            mc = MRIConvert()
            mc.inputs.in_file = os.path.join(sub, 'recon_all', 'mri', 'brain.mgz')
            mc.inputs.out_file = os.path.join(sub, 'recon_all', 'mri', 'brain.nii.gz')
            mc.inputs.out_type = 'nii'
            mc.run()

            mc = MRIConvert()
            mc.inputs.in_file = os.path.join(sub, 'recon_all', 'mri', 'aseg.mgz')
            mc.inputs.out_file = os.path.join(sub, 'recon_all', 'mri', 'aseg.nii.gz')
            mc.inputs.out_type = 'nii'
            mc.run()

            # registration to DCE space
            flt = FLIRT(bins=256, cost='corratio',
                        in_file=os.path.join(sub, 'recon_all', 'mri', 'brain.nii.gz'),
                        reference=os.path.join(sub, 'DCE', 'DCE4D_vol0002_brain.nii.gz'), output_type='NIFTI_GZ',
                        out_file=os.path.join(sub, 'recon_all', 'mri', 'r2DCE_brain.nii.gz'),
                        out_matrix_file=os.path.join(sub, 'DCE', 'r2DCEbrain.mat'), searchr_x=[-180, 180],
                        searchr_y=[-180, 180], searchr_z=[-180, 180],
                        dof=6,
                        interp='trilinear')
            flt.run()

            flt = fsl.FLIRT(bins=256, cost='corratio',
                            in_file=os.path.join(sub, 'recon_all', 'mri', 'aseg.nii.gz'),
                            reference=os.path.join(sub, 'DCE', 'DCE4D_vol0002_brain.nii.gz'),
                            output_type='NIFTI_GZ',
                            out_file=os.path.join(sub, 'recon_all', 'mri', 'r2DCE_aseg.nii.gz'),
                            apply_xfm=True,
                            in_matrix_file=os.path.join(sub, 'DCE', 'r2DCEbrain.mat'), interp='trilinear')
            flt.run()
            for key, value in regions.items():
                create_dce_mask(sub, key, value)
                calc_Ktrans2_in_mask(sub, value)
            # organize txt files:
            for key, value in regions.items():
                dce_summarize_results(sub, value, 'Ktrans2N_no_thr')
                dce_summarize_volumes(sub, value)
                dce_summarize_sub_volumes(sub, value, 'Ktrans2N_no_thr')


def dti_summarize_results(direct, region_name, dti_map):
    """
    The method gets region name and dti_map name and summarize all the results in the current direct to one united
    txt file direct is the direct of the DTI files (for example: new_healthyControl/DTI)
    """
    with open(os.path.join(''.join([dti_map, '_median_', region_name, '_.txt'])), 'w') as outfile:
        for sub in find_sub_dirs(direct, path_avoid):
            with open(os.path.join(sub, 'DTI', dti_map,
                                   ''.join(['median_', dti_map, '_', region_name, '.txt']))) as infile:
                outfile.write(infile.read())

    with open(os.path.join(''.join([dti_map, '_mean_', region_name, '_.txt'])), 'w') as outfile:
        for sub in find_sub_dirs(direct, path_avoid):
            with open(os.path.join(sub, 'DTI', dti_map,
                                   ''.join(['_mean_', dti_map, '_', region_name, '.txt']))) as infile:
                outfile.write(infile.read())


def dti_summarize_volumes(direct, region_name, dti_map):
    with open(os.path.join(direct, 'DTI', dti_map, 'volume_', region_name, '_.txt'),
              'w') as outfile:
        for sub in find_sub_dirs(direct, path_avoid):
            with open(
                    os.path.join(sub, 'DTI', dti_map, region_name, '_cl_bin_mask.txt')) as infile:
                outfile.write(infile.read())


def create_FA_WM_mask(direct, thresh=0.25):
    thr = Threshold()
    thr.inputs.in_file = os.path.join(direct, 'output', 'r2FS_FA.nii.gz')
    thr.inputs.thresh = thresh
    thr.inputs.out_file = os.path.join(direct, 'output', 'r2FS_FA_WM_mask.nii.gz')
    thr.run()

    # Convert to bin mask:
    bin = UnaryMaths()
    bin.inputs.in_file = os.path.join(direct, 'output', 'r2FS_FA_WM_mask.nii.gz')
    bin.inputs.operation = 'bin'
    bin.inputs.out_file = os.path.join(direct, 'output', 'r2FS_FA_WM_bin_mask.nii.gz')
    bin.run()


def create_MD_csf_mask(direct, thresh=0.001):
    thr = Threshold()
    thr.inputs.in_file = os.path.join(direct, 'output', 'r2FS_MD.nii.gz')
    thr.inputs.thresh = thresh
    thr.inputs.direction = 'above'
    thr.inputs.out_file = os.path.join(direct, 'output', 'r2FS_MD_csf_mask.nii.gz')
    thr.run()

    # Convert to bin mask:
    bin = UnaryMaths()
    bin.inputs.in_file = os.path.join(direct, 'output', 'r2FS_MD_csf_mask.nii.gz')
    bin.inputs.operation = 'bin'
    bin.inputs.out_file = os.path.join(direct, 'output', 'r2FS_MD_csf_bin_mask.nii.gz')
    bin.run()


def create_dti_mask(direct, region_number, region_name):
    """
    The method gets a direct,  region_number and region_name and
    creates masks of the given region registered to FreeSurfer space
    """
    region_mask = os.path.join(direct, 'recon_all', 'mri', 'aseg.nii.gz')
    region_mask_img = nib.load(region_mask)
    region_mask_data = region_mask_img.get_fdata()
    region_mask_data[region_mask_data != int(region_number)] = 0
    new_region_mask = nib.Nifti1Image(region_mask_data, region_mask_img.affine, region_mask_img.header)
    nib.save(new_region_mask, os.path.join(direct, 'recon_all', 'mri', ''.join(['r2FS_', region_name, '_mask.nii.gz'])))

    cl = Cluster()
    cl.inputs.in_file = os.path.join(os.path.join(direct, 'recon_all', 'mri', ''.join(['r2FS_', region_name, '_mask'
                                                                                                             '.nii.gz'])))
    cl.inputs.threshold = int(region_number)  # arbitrary threshold
    cl.inputs.out_size_file = os.path.join(direct, 'recon_all', 'mri',
                                           ''.join(['r2FS_', region_name, '_cl_mask.nii.gz']))
    cl.run()

    thr = Threshold()
    thr.inputs.in_file = os.path.join(direct, 'recon_all', 'mri', ''.join(['r2FS_', region_name, '_cl_mask.nii.gz']))
    thr.inputs.thresh = 5
    thr.inputs.out_file = os.path.join(direct, 'recon_all', 'mri', ''.join(['r2FS_', region_name, '_cl_mask.nii.gz']))
    thr.run()

    # Convert to bin mask:
    binary = UnaryMaths()
    binary.inputs.in_file = os.path.join(direct, 'recon_all', 'mri', ''.join(['r2FS_', region_name, '_cl_mask.nii.gz']))
    binary.inputs.operation = 'bin'
    binary.inputs.out_file = os.path.join(direct, 'recon_all', 'mri',
                                          ''.join(['r2FS_', region_name, '_cl_mask.nii.gz']))
    binary.run()


def create_WM_mask(direct, region_name):
    """
    The method gets a direct,  region_number and region_name and
    creates masks of the given region WM (FA>0.25) registered to FreeSurfer space
    """
    WM_mask = BinaryMaths()
    WM_mask.inputs.in_file = os.path.join(direct, 'recon_all', 'mri',
                                          ''.join(['r2FS_', region_name, '_cl_mask.nii.gz']))
    WM_mask.inputs.operation = 'mul'
    WM_mask.inputs.operand_file = os.path.join(direct, 'output', 'r2FS_FA_WM_bin_mask.nii.gz')
    WM_mask.inputs.nan2zeros = True
    WM_mask.inputs.out_file = os.path.join(direct, 'recon_all', 'mri',
                                           ''.join(['r2FS_', region_name, '_FA_WM_bin_mask.nii.gz']))
    WM_mask.run()


def create_no_CSF_mask(direct, region_name):
    """
    The method gets a direct,  region_number and region_name and
    creates masks of the given region without CSF (MD<0.001) registered to FreeSurfer space
    """
    no_csf_mask = BinaryMaths()
    no_csf_mask.inputs.in_file = os.path.join(direct, 'recon_all', 'mri',
                                              ''.join(['r2FS_', region_name, '_cl_mask.nii.gz']))
    no_csf_mask.inputs.operation = 'mul'
    no_csf_mask.inputs.operand_file = os.path.join(direct, 'output', 'r2FS_MD_csf_bin_mask.nii.gz')
    no_csf_mask.inputs.nan2zeros = True
    no_csf_mask.inputs.out_file = os.path.join(direct, 'recon_all', 'mri',
                                               ''.join(['r2FS_', region_name, '_MD_csf_bin_mask.nii.gz']))
    no_csf_mask.run()


def calc_dti_in_mask(direct, region_name, dti_map, mask_file):
    '''
    The method gets a directory and region_name and a threshold and calculates Ktrans2N in the given region by the folowing steps:
        1. Multiply Ktrans2N map by the given region binary mask (creates Ktrans2N_region_name.nii.gz file)
        2. Convert to bin mask
        3. Extract volume (in voxels and mm^3) in the whole region and in the area that been obtained in section 4.
        4. Calculates the median value of Ktrans2N in the area that been obtained in section 4.
    '''
    """
    The method gets a direct and region_name and calculates FA and MD in the given region:
    """

    # Extract volume (in voxels and mm^3) in the whole region:
    all_volume = ImageStats()
    all_volume.inputs.in_file = os.path.join(direct, 'recon_all', 'mri', ''.join(['r2FS_', region_name, '_cl_mask.nii'
                                                                                                        '.gz']))
    all_volume.inputs.op_string = '-V'
    all_volume.terminal_output = 'file'
    all_volume.run()
    os.rename("output.nipype",
              os.path.join(direct, 'DTI', dti_map, ''.join([region_name, '_cl_bin_mask.txt'])))

    # Calculate median FA value:
    median = ImageStats()
    median.inputs.in_file = os.path.join(direct, 'output', ''.join(['r2FS_', dti_map, '.nii.gz']))
    median.terminal_output = 'file'
    median.inputs.mask_file = os.path.join(direct, 'recon_all', 'mri', ''.join(['r2FS_', region_name, mask_file]))
    median.inputs.op_string = '-k %s -P 50 '
    median.run()
    os.rename("output.nipype",
              os.path.join(direct, 'DTI', dti_map, ''.join(['median_', dti_map, '_', region_name, '.txt'])))

    # Calculate mean FA value:
    mean = ImageStats()
    mean.inputs.in_file = os.path.join(direct, 'output', ''.join(['r2FS_', dti_map, '.nii.gz']))
    mean.terminal_output = 'file'
    mean.inputs.mask_file = os.path.join(direct, 'recon_all', 'mri', ''.join(['r2FS_', region_name, mask_file]))
    mean.inputs.op_string = '-k %s -M '
    mean.run()
    os.rename("output.nipype",
              os.path.join(direct, 'DTI', dti_map, ''.join(['mean_', dti_map, '_', region_name, '.txt'])))


def get_dti_vals(subjects_dir):
    for sub in find_sub_dirs(subjects_dir, path_avoid):
        print(sub)
        try:
            os.mkdir(os.path.join(sub, 'DTI'))
        except FileExistsError:
            pass
        try:
            os.mkdir(os.path.join(sub, 'DTI', 'FA'))
        except FileExistsError:
            pass
        try:
            os.mkdir(os.path.join(sub, 'DTI', 'MD'))
        except FileExistsError:
            pass
        '''
            #brain extraction to FA and MD maps'''
        perform_brain_extraction(os.path.join(sub, 'output'), 'DTI__MD.nii.gz', 0.15)
        perform_brain_extraction(os.path.join(sub, 'output'), 'DTI__FA.nii.gz', 0.15)
        if not os.listdir(os.path.join(sub, 'DTI', 'FA')) and not os.listdir(os.path.join(sub, 'DTI', 'MD')):
            # converting brain.mgz and aseg.mgz to nii files:

            mc = MRIConvert()
            mc.inputs.in_file = os.path.join(sub, 'recon_all', 'mri', 'brain.mgz')
            mc.inputs.out_file = os.path.join(sub, 'recon_all', 'mri', 'brain.nii.gz')
            mc.inputs.out_type = 'nii'
            mc.run()

            mc = MRIConvert()
            mc.inputs.in_file = os.path.join(sub, 'recon_all', 'mri', 'aseg.mgz')
            mc.inputs.out_file = os.path.join(sub, 'recon_all', 'mri', 'aseg.nii.gz')
            mc.inputs.out_type = 'nii'
            mc.run()

            # registration of FA & MD to freesurfer space
            flt = fsl.FLIRT(bins=256, cost='corratio',
                            in_file=os.path.join(sub, 'output', 'brain_DTI__FA.nii.gz'),
                            reference=os.path.join(sub, 'recon_all', 'mri',
                                                   'brain.nii.gz'),
                            output_type='NIFTI_GZ',
                            out_file=os.path.join(sub, 'output', 'r2FS_FA.nii.gz'),
                            out_matrix_file=os.path.join(sub, 'r2FS_FA.mat'), searchr_x=[-90, 90],
                            searchr_y=[-90, 90], searchr_z=[-90, 90], dof=12, interp='trilinear')
            flt.run()

            flt = fsl.FLIRT(bins=256, cost='corratio',
                            in_file=os.path.join(sub, 'output', 'brain_DTI__MD.nii.gz'),
                            reference=os.path.join(sub, 'recon_all', 'mri',
                                                   'brain.nii.gz'),
                            output_type='NIFTI_GZ',
                            out_file=os.path.join(sub, 'output', 'r2FS_MD.nii.gz'),
                            apply_xfm=True,
                            in_matrix_file=os.path.join(sub, 'r2FS_FA.mat'), interp='trilinear')
            flt.run()
            for key, value in regions.items():
                create_dti_mask(sub, key, value)
                create_FA_WM_mask(sub)
                create_MD_csf_mask(sub)
                create_WM_mask(sub, value)
                create_no_CSF_mask(sub, value)
                calc_dti_in_mask(sub, value, 'FA', '_FA_WM_bin_mask.nii.gz')
                calc_dti_in_mask(sub, value, 'MD', '_MD_csf_bin_mask.nii.gz')
    for key, value in regions.items():
        dti_summarize_results(subjects_dir, value, 'FA')
        dti_summarize_results(subjects_dir, value, 'MD')
        dti_summarize_volumes(subjects_dir, value, 'FA')
        dti_summarize_volumes(subjects_dir, value, 'MD')


def main():
    DTI(subjects_direct)
    #TBSS(subjects_direct)
    #recon(subjects_direct)
    # DCE(subjects_direct)
    # get_dti_vals(subjects_direct)
    # stats(tbss_direct, non_fa_direct)


# extract_values(stat_dir)

if __name__ == '__main__':
    main()
