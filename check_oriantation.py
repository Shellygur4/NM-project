import os
import nibabel as nib

path = '/mnt/z/Biogen_Shelly/biogen_first_time_point_shelly/'
patients = os.listdir(path)
change_to_LPI_files = ["star_L_NC.nii.gz", "star_R_NC.nii.gz", "star_L_SN.nii.gz", "star_R_SN.nii.gz"]
i=0

for patient in patients:
    if 'full_parse' in patient:
        continue
    patient_path = os.path.join(path, patient)
    # for file in change_to_LPI_files:
    star_map2 = nib.load(os.path.join(patient_path, 'star_map2.nii'))
    # try:
    NM = nib.load(os.path.join(patient_path, 'NM.nii'))
    # except:
    #     continue

    affine_star_map2 = star_map2.affine
    affine_NM = NM.affine

    orientation_star_map2 = nib.orientations.aff2axcodes(affine_star_map2)
    orientation_NM = nib.orientations.aff2axcodes(affine_NM)

    print(patient)
    # print(file)
    print("Orientation star_map2:" ,orientation_star_map2)
    print("Orientation NM:" ,orientation_NM)
    
    if orientation_star_map2 != orientation_NM:
        i = i+1
        print('NOT THE SAME')
    break

print(i)

