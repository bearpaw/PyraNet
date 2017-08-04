#!/usr/bin/env sh
expID=mpii/mpii_hg_s8_r1_v3_bn_preact_useconv2
expID=mpii/mpii_hg_s8_r1_v511_scale4_aug
# expID=ablation_scales/mpii_hg_s2_v503_scale8
expID=mpii/mpii_hg_s8_r1_v511_scale4
dataset=mpii
gpuID=2,3
nGPU=2
batchSize=6
LR=2.5e-4
netType=hg_v4
nStack=8
nResidual=1
nThreads=2
minusMean=true
nClasses=16
nEpochs=250
epochNumber=250

cd ../..

CUDA_VISIBLE_DEVICES=$gpuID th main.lua \
	-dataset $dataset \
	-expID $expID \
	-batchSize $batchSize \
	-nGPU $nGPU \
	-LR $LR \
	-momentum 0.0 \
	-weightDecay 0.0 \
	-netType $netType \
	-nStack $nStack \
	-nResidual $nResidual \
	-nThreads $nThreads \
	-minusMean $minusMean \
   -nClasses $nClasses \
   -nEpochs $nEpochs \
   -epochNumber $epochNumber \
   -scales 0 \
   -loadModel checkpoints/$expID/model_$epochNumber.t7 -testOnly true #-testRelease true #-debug true
   # -loadModel /home/wyang/code/pose/umich-stacked-hourglass/umich-stacked-hourglass.t7 -testOnly true
