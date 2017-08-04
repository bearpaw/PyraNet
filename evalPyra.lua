--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'xlua'
local DataLoader = require 'dataloader-pyra'
local models = require 'models.init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'
local Logger = require 'utils.Logger'
-- local Initializer = require 'utils.weight-init'

local scales = torch.range(0.8, 1.3, 0.1):totable() 

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)


-- Data loading
local trainLoader, valLoader, testLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testRelease then
   print('=> Test Release')
   local testAcc, testLoss = trainer:multiScaleTest(opt.epochNumber, testLoader, scales)
   print(string.format(' * Results acc: %6.3f, loss: %6.3f', testAcc, testLoss))
   return
end

if opt.testOnly then
   print('=> Test Only')
   local testAcc, testLoss = trainer:multiScaleTest(opt.epochNumber, valLoader, scales)
   print(string.format(' * Results acc: %6.3f, loss: %6.3f', testAcc, testLoss))
   return
end
