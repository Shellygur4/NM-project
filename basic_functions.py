import glob
import os
import shutil
from nipype.interfaces import fsl
from nipype.interfaces.dcm2nii import Dcm2niix

def find_sub_dirs(directory, avoid=None):
    """
    Returns a list of subdirectories within the given directory, excluding any specified in the avoid list.
    """
    if avoid is None:
        avoid = ['']
    os.chdir(directory)
    sub_directory_names = glob.glob("*/")  # List only the directories
    sub_dirs = []
    for name in sub_directory_names:
        if name not in avoid:
            sub_dirs.append(os.path.join(directory, name))
    return sub_dirs

def remove_files(directory, name_list):
    """
    Removes directories or files specified in name_list within the given directory.
    """
    for name in name_list:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

def convert_dicom_to_nii(directory):
    """
    Converts DICOM directories to NIfTI images within the given directory using Dcm2niix.
    """
    converter = Dcm2niix(bids_format=False, compress='n', source_dir=directory, out_filename='%f')
    converter.run()

def perform_brain_extraction(directory, filename, frac=0.5):
    """
    Performs brain extraction on the given filename within the provided directory, creating skull-stripped brain masks.
    """
    btr = fsl.BET(in_file=os.path.join(directory, filename),
                  frac=frac,
                  out_file=os.path.join(directory, f'brain_{filename}'),
                  functional=True)
    btr.run()
