--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Training script for "Learning Feature Pyramids for Human Pose Estimation"')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-data',         '',         'Path to dataset')
   cmd:option('-dataset',      'mpii',     'Options: mpii | mpii-lsp')
   cmd:option('-manualSeed',   0,          'Manually set RNG seed')
   cmd:option('-nGPU',         1,          'Number of GPUs to use by default')
   cmd:option('-backend',      'cudnn',    'Options: cudnn | cunn')
   cmd:option('-cudnn',        'fastest',  'Options: fastest | default | deterministic')
   cmd:option('-gen',          'gen',      'Path to save generated files')
   ------------- Data options ------------------------
   cmd:option('-nThreads',     2,          'number of data loading threads')
   cmd:option('-inputRes',     256,        'Input image resolution')
   cmd:option('-outputRes',    64,         'Output heatmap resolution')
   cmd:option('-scaleFactor',  .25,        'Degree of scale augmentation')
   cmd:option('-rotFactor',    30,         'Degree of rotation augmentation')
   cmd:option('-rotProbab',    .4,         'Degree of rotation augmentation')
   cmd:option('-flipFactor',   .5,         'Degree of flip augmentation')
   cmd:option('-minusMean',    'true',     'Minus image mean')
   cmd:option('-gsize',        1,          'Kernel size to generate the Gassian-like labelmap')
   ------------- Training options --------------------
   cmd:option('-nEpochs',      0,          'Number of total epochs to run')
   cmd:option('-epochNumber',  1,          'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',    1,          'mini-batch size (1 = pure stochastic)')
   cmd:option('-testOnly',     'false',    'Run on validation set only')
   cmd:option('-testRelease',  'false',    'Run on testing set only')
   cmd:option('-crit',         'MSE',      'Criterion type: MSE | CrossEntropy')
   cmd:option('-optMethod',    'rmsprop',  'Optimization method: rmsprop | sgd | nag | adadelta | adam')
   cmd:option('-snapshot',     1,          'How often to take a snapshot of the model (0 = never)')
   cmd:option('-debug',        'false',    'Visualize training/testing samples')
   ------------- Checkpointing options ---------------
   cmd:option('-save',         'checkpoints','Directory in which to save checkpoints')
   cmd:option('-expID',        'default',  'Experiment ID')
   cmd:option('-resume',       'none',     'Resume from the latest checkpoint in this directory')
   cmd:option('-loadModel',    'none',     'Load model')  
   ---------- Optimization options ----------------------
   cmd:option('-LR',           2.5e-4,     'initial learning rate')
   cmd:option('-momentum',     0.0,        'momentum')
   cmd:option('-weightDecay',  0.0,        'weight decay')
   cmd:option('-alpha',        0.99,       'Alpha')
   cmd:option('-epsilon',      1e-8,       'Epsilon')
   cmd:option('-dropout',      0,          'Dropout ratio')
   cmd:option('-init',         'none',     'Weight initialization method: none | heuristic | xavier | xavier_caffe | kaiming')
   ---------- Model options ----------------------------------
   cmd:option('-netType',      'hg-prm',   'Options: hg-prm')
   cmd:option('-shortcutType', '',         'Options: A | B | C')
   cmd:option('-retrain',      'none',     'Path to model to retrain with')
   cmd:option('-optimState',   'none',     'Path to an optimState to reload from')
   cmd:option('-nStack',       8,          'Number of stacks in the provided hourglass model (for hg-generic)')
   cmd:option('-nFeats',       256,        'Number of features in the hourglass (for hg-generic)')
   cmd:option('-nResidual',    1,          'Number of residual module in the hourglass (for hg-generic)')
   cmd:option('-baseWidth',    6,          'PRM: base width', 'number')
   cmd:option('-cardinality',  30,         'PRM: cardinality', 'number')
   ---------- Model options ----------------------------------
   cmd:option('-shareGradInput','false',   'Share gradInput tensors to reduce memory usage')
   cmd:option('-optnet',       'false',    'Use optnet to reduce memory usage')
   cmd:option('-resetClassifier','false',  'Reset the fully connected layer for fine-tuning')
   cmd:option('-nClasses',     16,         'Number of classes in the dataset')
   cmd:option('-bg',           'false',    'If true, we will have an additional fg/bg labelmap')
   cmd:text()

   local opt = cmd:parse(arg or {})

   opt.testOnly = opt.testOnly ~= 'false'
   opt.testRelease = opt.testRelease ~= 'false'
   opt.tenCrop = opt.tenCrop ~= 'false'
   opt.shareGradInput = opt.shareGradInput ~= 'false'
   opt.optnet = opt.optnet ~= 'false'
   opt.resetClassifier = opt.resetClassifier ~= 'false'

   if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
      cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
   end

   if string.find(opt.dataset, 'mpii') ~= nil  or opt.dataset == 'lsp' then      
      -- Default shortcutType=B
      opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 200 or opt.nEpochs
   else
      cmd:error('unknown dataset: ' .. opt.dataset)
   end

   if opt.resetClassifier then
      if opt.nClasses == 0 then
         cmd:error('-nClasses required when resetClassifier is set')
      end
   end

   if opt.shareGradInput and opt.optnet then
      cmd:error('error: cannot use both -shareGradInput and -optnet')
   end

   return opt
end

return M
