import SimpleITK as sitk
import torch


mapping_visual={0:"background",1: 'adrenal_gland_left', 2: 'adrenal_gland_right', 3: 'aorta', 4: 'autochthon_left', 5: 'autochthon_right', 6: 'brain', 7: 'clavicula_left', 8: 'clavicula_right', 9: 'colon', 10: 'duodenum', 11: 'esophagus', 12: 'face', 13: 'femur_left', 14: 'femur_right', 15: 'gallbladder', 16: 'gluteus_maximus_left', 17: 'gluteus_maximus_right', 18: 'gluteus_medius_left', 19: 'gluteus_medius_right', 20: 'gluteus_minimus_left', 21: 'gluteus_minimus_right', 22: 'heart_atrium_left', 23: 'heart_atrium_right', 24: 'heart_myocardium', 25: 'heart_ventricle_left', 26: 'heart_ventricle_right', 27: 'hip_left', 28: 'hip_right', 29: 'humerus_left', 30: 'humerus_right', 31: 'iliac_artery_left', 32: 'iliac_artery_right', 33: 'iliac_vena_left', 34: 'iliac_vena_right', 35: 'iliopsoas_left', 36: 'iliopsoas_right', 37: 'inferior_vena_cava', 38: 'kidney_left', 39: 'kidney_right', 40: 'liver', 41: 'lung_lower_lobe_left', 42: 'lung_lower_lobe_right', 43: 'lung_middle_lobe_right', 44: 'lung_upper_lobe_left', 45: 'lung_upper_lobe_right', 46: 'pancreas', 47: 'portal_vein_and_splenic_vein', 48: 'pulmonary_artery', 49: 'rib_left_1', 50: 'rib_left_2', 51: 'rib_left_3', 52: 'rib_left_4', 53: 'rib_left_5', 54: 'rib_left_6', 55: 'rib_left_7', 56: 'rib_left_8', 57: 'rib_left_9', 58: 'rib_left_10', 59: 'rib_left_11', 60: 'rib_left_12', 61: 'rib_right_1', 62: 'rib_right_2', 63: 'rib_right_3', 64: 'rib_right_4', 65: 'rib_right_5', 66: 'rib_right_6', 67: 'rib_right_7', 68: 'rib_right_8', 69: 'rib_right_9', 70: 'rib_right_10', 71: 'rib_right_11', 72: 'rib_right_12', 73: 'sacrum', 74: 'scapula_left', 75: 'scapula_right', 76: 'small_bowel', 77: 'spleen', 78: 'stomach', 79: 'trachea', 80: 'urinary_bladder', 81: 'vertebrae_C1', 82: 'vertebrae_C2', 83: 'vertebrae_C3', 84: 'vertebrae_C4', 85: 'vertebrae_C5', 86: 'vertebrae_C6', 87: 'vertebrae_C7', 88: 'vertebrae_L1', 89: 'vertebrae_L2', 90: 'vertebrae_L3', 91: 'vertebrae_L4', 92: 'vertebrae_L5', 93: 'vertebrae_T1', 94: 'vertebrae_T2', 95: 'vertebrae_T3', 96: 'vertebrae_T4', 97: 'vertebrae_T5', 98: 'vertebrae_T6', 99: 'vertebrae_T7', 100: 'vertebrae_T8', 101: 'vertebrae_T9', 102: 'vertebrae_T10', 103: 'vertebrae_T11', 104: 'vertebrae_T12'}

hierarchy_mapping = {
    "adrenal_glands":[1,2],
    "autochthon muscles":[4,5],
    "clavicles":[7,8],
    "femurs":[13,14],
    "gluteus muscles":[16,17,18,19,20,21],
    "hips":[27,28],
    "humerus":[29,30],
    "iliac arteries":[31,32],
    "iliac veins":[33,34],
    "iliopsoas muscles":[35,36],
    "kidneys":[38,39],
    "lungs":[41,42,43,44,45],
    "scapulas":[74,75],
    "vertebrae_C":[81,82,83,84,85,86,87],
    "vertebrae_T":[93,94,95,96,97,98,99,100,101,102,103,104],
    "vertebrae_L":[88,89,90,91,92],
    "heart":[22,23,24,25,26],
    
    "brain and face":[6,12], # this could be useful for neurocranial studies
    "vessels":[31,32,48,33,34,37,47,3],
    "abdominal organs":[40,15,38,39,46,77,78,76,9,80],
    "thoracic organs":[22,23,24,25,26,41,42,43,44,45,11],
    "organs":[1,2,6,9,10,11,15,38,39,40,41,42,43,44,45,46,76,77,78,79,80,12],
    "vertebraes":[81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104],
    "muscles":[4,5,16,17,18,19,20,21,35,36],
    "ribs":[49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72],
    "bones":[7,8,13,14,27,28,29,30,73,74,75,81,82,83,84,85,86,87,88,
             89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,49,50,51,52,53,54,
             55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72],
    
    "cardiovascular system":[3,22,23,24,25,26,31,32,33,34,37,47,48],
    "gastrointestinal tract":[9,10,11,76,78], # 
    "respiratory system":[41,42,43,44,45,79], # 
    "urinary system":[38,39,80],
    "digestive system":[11,78,76,9,15,46],
    "musculoskeletal system":[4,5,16,17,18,19,20,21,35,36,7,8,13,14,27,28,29,
                              30,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,
                              64,65,66,67,68,69,70,71,72,73,74,75,81,82,83,84,85,86,
                              87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104],

}

BTCV_cervix_mapping = {0:0,1:80,2:0,3:0,4:76}
msd_spleen_mapping = {0:0,1:77}
segthor_mapping = {0:0,1:1,2:121,3:79,4:3}



abdomen1k_mapping = None

hierachy = None
