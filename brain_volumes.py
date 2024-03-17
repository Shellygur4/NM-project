import numpy as np
import nibabel as nb

#Shahar script
#get volume of all brain
def get_nonzero_vol(INPUT):
    # Load data
    nii = nb.load(INPUT)
    img = nii.get_fdata()
    # Get voxel dimensions
    voxel_dims = (nii.header["pixdim"])[1:4]
    print("Voxel dimensions:")
    print("  x = {} mm".format(voxel_dims[0]))
    print("  y = {} mm".format(voxel_dims[1]))
    print("  z = {} mm".format(voxel_dims[2]))
    # Compute volume
    nonzero_voxel_count = np.count_nonzero(img)
    voxel_volume = np.prod(voxel_dims)
    nonzero_voxel_volume = nonzero_voxel_count * voxel_volume
    print("Number of non-zero voxels = {}".format(nonzero_voxel_count))
    print("Volume of non-zero voxels (all brain) = {} mm^3".format(nonzero_voxel_volume))
    return nonzero_voxel_volume

INPUT = "C:/Shahar/shahar/Data_ni/A_R_1/Study20160610_074148/T2_noskull.nii"
brain_vol = get_nonzero_vol(INPUT)

#get the volume of lesions
def get_lesions_vol(lesion_path):
    lesions = nb.load(lesion_path)
    # Get voxel dimensions
    voxel_dims = (lesions.header["pixdim"])[1:4]
    print("Voxel dimensions:")
    print("  x = {} mm".format(voxel_dims[0]))
    print("  y = {} mm".format(voxel_dims[1]))
    print("  z = {} mm".format(voxel_dims[2]))
    # Compute volume
    voxel_volume = np.prod(voxel_dims)
    lesion_mask = lesions.get_fdata()
    lesion_voxel_count = np.count_nonzero(lesion_mask)
    lesion_volume = voxel_volume * lesion_voxel_count
    print("Volume of lesions = {} mm^3".format(lesion_volume))
    return lesion_volume

lesion_path = "C:/Shahar/LABELS/A_R_1/Study20160610_074148/correction.nii"
les_vol = get_lesions_vol(lesion_path)

print("the volume of the brain is: " + str(brain_vol) + ",and the lesion volume is: " + str(les_vol))