#!/usr/bin/env sh
expID=mpii/hg-prm-stack2
dataset=mpii
gpuID=0
nGPU=1
batchSize=6
LR=2.5e-4
netType=hg-prm
nStack=2
nResidual=1
nThreads=8
minusMean=true
nClasses=16
nEpochs=250
snapshot=10
nFeats=256
baseWidth=9
cardinality=4

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
	-snapshot $snapshot \
	-nFeats $nFeats \
	-baseWidth $baseWidth \
	-cardinality $cardinality \
	# -resume checkpoints/$expID
