#!/bin/bash

#Linear registration of masks 
#from MNI152 standard space to T1w subject space
#and from T1w subject space to bold3Tp2 subject space

maskdir=$1
transformsdir=$2
outdir=$3

for sub in 01 02 03 04 05 06 09 10 14 15 16 17 18 19 20; do

	echo "registering $(basename $maskdir) to T1w subject $sub space"
	/usr/local/fsl/bin/flirt \
		-in $maskdir \
		-applyxfm \
		-init ${transformsdir}/sub-${sub}/t1w/in_mni152/tmpl2subj.mat \
		-out ${outdir}/$(basename $maskdir)_sub-${sub}_to_t1w.nii \
		-paddingsize 0.0 \
		-interp trilinear \
		-ref ${transformsdir}/sub-${sub}/t1w/brain.nii.gz;
		
	echo "registering $(basename $maskdir) to bold3Tp2 subject $sub space"
	/usr/local/fsl/bin/flirt \
		-in ${outdir}/$(basename $maskdir)_sub-${sub}_to_t1w.nii \
		-applyxfm \
		-init ${transformsdir}/sub-${sub}/t1w/in_bold3Tp2/xfm_6dof.mat \
		-out ${outdir}/$(basename $maskdir)_sub-${sub}_to_bold3Tp2.nii \
		-paddingsize 0.0 \
		-interp trilinear \
		-ref ${transformsdir}/sub-${sub}/bold3Tp2/brain.nii.gz;
done
