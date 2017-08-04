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
local DataLoader = require 'dataloader'
local models = require 'models.init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'
local Logger = require 'utils.Logger'

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
   local testAcc, testLoss = trainer:test(opt.epochNumber, testLoader)
   print(string.format(' * Results acc: %6.3f, loss: %6.3f', testAcc, testLoss))
   return
end

if opt.testOnly then
   print('=> Test Only')
   local testAcc, testLoss = trainer:test(opt.epochNumber, valLoader)
   print(string.format(' * Results acc: %6.3f, loss: %6.3f', testAcc, testLoss))
   return
end

local r_step, d_step = 3/opt.nEpochs, 5/opt.nEpochs

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestAcc = -math.huge
local bestEpoch = 0
local logger = Logger(paths.concat(opt.save, opt.expID, 'full.log'), opt.resume ~= 'none')
logger:setNames{'Train acc.', 'Train loss.', 'Test acc.', 'Test loss.'}
logger:style{'+-', '+-', '+-', '+-'}
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainAcc, trainLoss = trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local testAcc, testLoss = trainer:test(epoch, valLoader)

   -- Write to logger
   logger:add{trainAcc, trainLoss, testAcc, testLoss}
   print((' Finished epoch # %d'):format(epoch))
   print(('\tTrain Loss: %.4f, Train Acc: %.4f'):format(trainLoss, trainAcc))
   print(('\tTest Loss:  %.4f, Test Acc:  %.4f'):format(testLoss, testAcc))

   local bestModel = false
   if testAcc > bestAcc then
      bestModel = true
      bestAcc = testAcc
      bestEpoch = epoch
      checkpoints.saveBest(epoch, model, opt)
      print(('\tBest model: %.4f [*]'):format(bestAcc))
   end

   if epoch % opt.snapshot == 0 then
      checkpoints.save(epoch, model, trainer.optimState, opt)
   end

   collectgarbage()
end

print(string.format(' * Finished acc: %6.3f, Best epoch: %d', bestAcc, bestEpoch))
