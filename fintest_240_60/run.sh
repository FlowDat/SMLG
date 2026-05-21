#!/bin/bash
#SBATCH --job-name=openfoam_simulation2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16


date 

blockMesh

surfaceFeatureExtract

#rm system/decomposeParDict
#cp system/decomposeParDict1 system/decomposeParDict

#decomposePar -force

#mpirun -np 8 snappyHexMesh -overwrite -parallel

snappyHexMesh -overwrite

#reconstructParMesh -constant

createPatch -overwrite

#rm system/decomposeParDict
#cp system/decomposeParDict2 system/decomposeParDict

checkMesh

date

decomposePar -force

mpirun -np 8 renumberMesh -overwrite -parallel

mpirun -np 8 icoFoam -parallel

date

mpirun -np 1 pvbatch pvdat.py

date

