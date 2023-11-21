#!/bin/bash

# Set the path to the FSL directory
FSLDIR=

# Directories
DATASET_DIR=

sub_counter=1
ds_counter=1

# Echo spacing array
declare -a ESP=()

for dataset_folder in ${DATASET_DIR}/ds*; do

  # effective echo spacing
  ESPi=${ESP[$((ds_counter - 1))]}

  for subject_folder in ${dataset_folder}/sub*; do

    ANAT_DIR="$subject_folder"/anat
    FMAP_DIR="$subject_folder"/fmap
    FUNC_DIR="$subject_folder"/func

    echo "Subject "$sub_counter" pre-processing"
    echo "------------------------"
    # 1. BET of T1w structural images
    echo "BET of T1w structural images..."
    ${FSLDIR}/bin/bet ${ANAT_DIR}/*T1w.nii.gz ${ANAT_DIR}/T1w_brain.nii.gz -f 0.1 -g 0 -m

    # 2. Prepare the field fmap
    echo "Field map prep..."
    # 2.1. BET of magnitude image
    ${FSLDIR}/bin/bet ${FMAP_DIR}/*magnitude1.nii.gz ${FMAP_DIR}/magnitude1_brain.nii.gz
    # 2.2. Erode magnitude1_brain
    ${FSLDIR}/bin/fslmaths ${FMAP_DIR}/magnitude1_brain.nii.gz -ero ${FMAP_DIR}/magnitude1_brain_ero.nii.gz
    # 2.3. Get the field map in rad/seconds
    ${FSLDIR}/bin/fsl_prepare_fieldmap SIEMENS ${FMAP_DIR}/*phasediff.nii.gz ${FMAP_DIR}/magnitude1_brain_ero.nii.gz ${FMAP_DIR}/fmap_rads.nii.gz 2.46

    # 3. Create example_func.nii.gz and example_func_10.nii.gz
    TOTAL_TF=$(fslnvols ${FUNC_DIR}/*bold.nii.gz)
    MEDIAN_TF=$((TOTAL_TF / 2))
    ${FSLDIR}/bin/fslroi ${FUNC_DIR}/*bold.nii.gz ${FUNC_DIR}/example_func.nii.gz ${MEDIAN_TF} 1
    ${FSLDIR}/bin/fslroi ${FUNC_DIR}/*bold.nii.gz ${FUNC_DIR}/example_func_10.nii.gz $((MEDIAN_TF - 5)) 10

    # 4. FUGUE correction
    echo "FUGUE correction..."
    ${FSLDIR}/bin/fugue -i ${FUNC_DIR}/example_func_10.nii.gz --dwell=${ESPi} --loadfmap=${FMAP_DIR}/fmap_rads.nii.gz --smooth3=3 --unwarpdir=y- -u ${FUNC_DIR}/example_func_10_FUGUE.nii.gz

    # 5. Registration to structural image
    # 5.1. Calculate the transformation matrix (EPI to structural)
    echo "Structural to EPI image registration..."
    # Optional: reorient the data to standard space
    ANAT_IMG=$(echo ${ANAT_DIR}/*T1w.nii.gz)
    ANAT_IMG_FILENAME=$(basename -- $ANAT_IMG)
    ${FSLDIR}/bin/fslreorient2std ${ANAT_DIR}/*T1w.nii.gz ${ANAT_DIR}/${ANAT_IMG_FILENAME}
    # Optional ends
    ${FSLDIR}/bin/epi_reg --epi=${FUNC_DIR}/example_func.nii.gz --t1=${ANAT_DIR}/*T1w.nii.gz --t1brain=${ANAT_DIR}/T1w_brain.nii.gz --out=${FUNC_DIR}/example_func2struct
    # 5.2. Invert the BOLD to structural transformation
    ${FSLDIR}/bin/convert_xfm -omat ${ANAT_DIR}/struct2example_func.mat -inverse ${FUNC_DIR}/example_func2struct.mat
    # 5.3. Apply the transformation to the structural image and T1w_brain_mask
    ${FSLDIR}/bin/flirt -in ${ANAT_DIR}/T1w_brain.nii.gz -ref ${FUNC_DIR}/example_func_10_brain.nii.gz -applyxfm -init ${ANAT_DIR}/struct2example_func.mat -out ${ANAT_DIR}/rT1w_brain.nii.gz
    ${FSLDIR}/bin/flirt -in ${ANAT_DIR}/T1w_brain_mask.nii.gz -ref ${FUNC_DIR}/example_func_10_brain.nii.gz -applyxfm -init ${ANAT_DIR}/struct2example_func.mat -out ${ANAT_DIR}/rT1w_brain_mask.nii.gz
    # 6. BET of the functional images
    # echo "BET of the functional images..."
    ${FSLDIR}/bin/bet ${FUNC_DIR}/rexample_func_10.nii.gz ${FUNC_DIR}/rexample_func_10_brain.nii.gz -F
    ${FSLDIR}/bin/bet ${FUNC_DIR}/rexample_func_10_FUGUE.nii.gz ${FUNC_DIR}/rexample_func_10_FUGUE_brain.nii.gz -F

    # echo "Subject "counter" done"

    sub_counter=$((sub_counter + 1))

  done

ds_counter=$((ds_counter + 1))

done
