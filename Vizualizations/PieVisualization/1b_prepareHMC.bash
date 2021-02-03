#!/bin/bash
cp -r ../../HMC/Code/Data/resdata .    #Copy results from HMC
cd ./resdata
#----------------------#
#--- Reshuffle data ---#
#----------------------#
#--- GP-LVM ---#
mkdir BGPLVM_NoReplace_14
mv rshfl_gpall* BGPLVM_NoReplace_14/.
mkdir BGPLVM_NoReplace
mv rshfl_gpred* BGPLVM_NoReplace/.
#--- Procrustes ---#
mkdir Procrustes_NoReplace
mv rshfl_prc* Procrustes_NoReplace/.
##----------------------#
##--- Resamples data ---#
##----------------------#
##--- GP-LVM ---#
#mkdir BGPLVM_replace_14
#mv rsmp_gpall* BGPLVM_replace_14/.
#mkdir BGPLVM_replace
#mv rsmp_gpred* BGPLVM_replace/.
##--- Procrustes ---#
#mkdir Procrustes_replace
#mv rsmp_prc* Procrustes_replace/.
#--------------------#
#--- Rename files ---#
#--------------------#
for i in {0..9}
do
    #--- GP-LVM ---#
	#--- top 14 ---#
    cp "./BGPLVM_NoReplace_14/rshfl_gpall_it""$i""__allardres.csv" "./BGPLVM_NoReplace_14/$i""_Relevance_.csv"
    #cp "./BGPLVM_replace_14/rsmp_gpall_it""$i""__allardres.csv"   "./BGPLVM_replace_14/$i""_Relevance.csv"
	#--- manual selected ---#
    cp "./BGPLVM_NoReplace/rshfl_gpred_it""$i""__allardres.csv" "./BGPLVM_NoReplace/$i""_Relevance_.csv"
    #cp "./BGPLVM_replace/rsmp_gpred_it""$i""__allardres.csv"   "./BGPLVM_replace/$i""_Relevance.csv"

    #--- Procrustes ---#
    cp "./Procrustes_NoReplace/rshfl_prc_it""$i""__allardres.csv" "./Procrustes_NoReplace/$i""_Relevance_.csv"
    #cp "./Procrustes_replace/rsmp_prc_it""$i""__allardres.csv"   "./Procrustes_replace/$i""_Relevance.csv"
done
