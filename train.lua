--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'
local hdf5 = require 'hdf5'
local Logger = require 'utils.Logger'
local Transformer = require 'datasets/posetransforms'
local nngraph = require 'nngraph'
local utils = require 'utils.utils'
local visualizer = require 'utils.visualizer'
local matio = require 'matio'
-- nngraph.setDebug(true)  -- uncomment for debugging nngraph

local M = {}
local Trainer = torch.class('resnet.Trainer', M)
 
-------------------------------------------------------------------------------
-- Helper Functions
-------------------------------------------------------------------------------
local shuffleLR = utils.shuffleLR
local flip = utils.flip
local applyFn = utils.applyFn
local finalPreds = utils.finalPreds

function Trainer:__init(model, criterion, opt, optimState)
   self.isDebug = opt.debug == 'true' or false -- If true, the generated train/val samples will be saved into tmp/ folder
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
      alpha = opt.alpha,
      epsilon = opt.epsilon
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
   print(('    Parameters: %.2fM'):format(self.params:size(1)/1000000))

   if opt.bg == 'true' then print('    Training with fg/bg map') end

   -- create logger
   self.trainLogger = Logger(paths.concat(opt.save, opt.expID, 'train.log'), opt.resume ~= 'none')
   self.testLogger = Logger(paths.concat(opt.save, opt.expID, 'test.log'), opt.resume ~= 'none')
   self.trainLogger:setNames{'epoch', 'iter', 'acc', 'loss', 'lr', 'minGrad', 'maxGrad', 'meanGrad', 'stdGrad'}
   self.testLogger:setNames{'epoch', 'iter', 'acc', 'loss'}
end

------------------------------------------------------------------------
--  Train
------------------------------------------------------------------------
function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local accSum, lossSum = 0.0, 0.0
   local N = 0
   local nIter = 0

   print('=> Training epoch # ' .. epoch)

   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      nIter = nIter + 1
      xlua.progress(n, trainSize)
      local dataTime = dataTimer:time().real
      if sample ~= nil then
         local im = sample.input
         -- Copy input and target to the GPU
         self:copyInputs(sample)

         local batchSize = self.input:size(1)
         local output = self.model:forward(self.input)
         local loss = self.criterion:forward(self.model.output, self.target)

         self.model:zeroGradParameters()
         self.criterion:backward(self.model.output, self.target)
         self.model:backward(self.input, self.criterion.gradInput)

         optim[self.opt.optMethod](feval, self.params, self.optimState)

         -- Calculate accuracy
         local acc = self:computeScore(output, self.target)

         lossSum = lossSum + loss*batchSize
         accSum = accSum + acc*batchSize
         N = N + batchSize

         print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.6f    Loss %.6f    Acc %.6f'):format(
            epoch, n, trainSize, timer:time().real, dataTime, loss, acc))

         self.trainLogger:add{epoch, n, acc, loss, self.optimState.learningRate, self.gradParams:min(), self.gradParams:max(), self.gradParams:mean(), self.gradParams:std()}
         -- check that the storage didn't get changed do to an unfortunate getParameters call
         assert(self.params:storage() == self.model:parameters()[1]:storage())

         if self.isDebug then
            local image = require('image')
            local gtIm = visualizer.drawOutput(self.input[1]:float(), self.target[#self.target][1]:float())
            local outIm = visualizer.drawOutput(self.input[1]:float(), self.model.output[#self.model.output][1]:float())
            win1=image.display{image=gtIm, win=win1, legend=('train gt: %d | %d'):format(n, trainSize)}
            win2=image.display{image=outIm, win=win2, legend=('train output: %d | %d'):format(n, trainSize)}
            sys.sleep(0.2)
         end

         timer:reset()
         dataTimer:reset()
      end
   end

   return accSum / N, lossSum / N
end

------------------------------------------------------------------------
--  Testing
------------------------------------------------------------------------
function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()
   local predsTable, indexTable = {}, {}

   local nCrops = self.opt.tenCrop and 10 or 1
   local lossSum, accSum = 0.0, 0.0
   local N = 0

   self.model:evaluate()
   for n, sample in dataloader:run(false) do
      xlua.progress(n, size)
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)
      local batchSize = self.input:size(1)

      -- Output of original image
      local output = self.model:forward(self.input)
      local loss = self.criterion:forward(self.model.output, self.target)

      -- Output of flipped image
      output = applyFn(function (x) return x:clone() end, output)
      local flippedOut = self.model:forward(flip(self.input))
      flippedOut = applyFn(function (x) return flip(shuffleLR(x, self.opt)) end, flippedOut)
      output = applyFn(function (x,y) return x:add(y):div(2) end, output, flippedOut)

      if self.isDebug then
         local image = require('image')
         local gtIm = visualizer.drawOutput(self.input[1]:float(), self.target[#self.target][1]:float())
         local outIm = visualizer.drawOutput(self.input[1]:float(), output[#output][1]:float())
         win1=image.display{image=gtIm, win=win1, legend=('Test gt: %d | %d'):format(n, size)}
         win2=image.display{image=outIm, win=win2, legend=('Test output: %d | %d'):format(n, size)}
         -- sys.sleep(0.5)
      end
      -- Get predictions (hm and img refer to the coordinate space)
      local preds_hm, preds_img = finalPreds(output[#output], sample.center, sample.scale)
      table.insert(predsTable, preds_img)
      table.insert(indexTable, sample.index)

      for ii = 1, sample.input:size(1) do
         matio.save(('checkpoints/mpii/hg-prm-stack2/results/valid_%.4d.mat'):format(sample.index[ii]),{image=sample.input[ii], score=output[#output]:float(), preds=preds_hm[ii]})
      end

      local acc, preds_hm = self:computeScore(output, sample.target)
      lossSum = lossSum + loss*batchSize
      accSum = accSum + acc*batchSize
      N = N + batchSize
      

      print((' Testing | Epoch: [%d][%d/%d]    Time %.3f  Data %.6f    Loss %.6f    Acc %.6f'):format(
       epoch, n, size, timer:time().real, dataTime, loss, acc))

      if epoch ~= 0 then
         self.testLogger:add{epoch, n, acc, loss}
      end

      timer:reset()
      dataTimer:reset()
   end
   self.model:training()

    -- Saving predictions
    local preds = torch.Tensor(N, predsTable[1]:size(2), predsTable[1]:size(3))
    local ptr = 1
    for k,v in pairs(predsTable) do
       local batchSize = v:size(1)
       for j = 1, batchSize do            
        local idx = indexTable[k][j]
        local pred = v[j]
        preds[idx]:copy(pred)
       end
    end
    local predFilePath = paths.concat(self.opt.save, self.opt.expID, 'pred_post_' .. epoch .. '.h5')
    local predFile = hdf5.open(predFilePath, 'w')
    predFile:write('preds', preds)
    predFile:close()

   return accSum / N, lossSum / N
end

------------------------------------------------------------------------
--  Multi-scale Testing
------------------------------------------------------------------------
function Trainer:multiScaleTest(epoch, dataloader, scales)

   -- Helper function: mapping heatmaps to original image size
   local function getHeatmaps(imHeight, imWidth, center, scale, rot, res, hm)
      local t = require 'datasets/posetransforms'
      local image = require('image')
      -- Crop function tailored to the needs of our system. Provide a center
      -- and scale value and the image will be cropped and resized to the output
      -- resolution determined by res. 'rot' will also rotate the image as needed.
      local ul = t.transform({1,1}, center, scale, 0, res, true)
      local br = t.transform({res,res}, center, scale, 0, res, true)

      local pad = math.floor(torch.norm((ul - br):float())/2 - (br[1]-ul[1])/2)
      if rot ~= 0 then
          ul = ul - pad
          br = br + pad
      end

      local newDim,newImg,ht,wd

      newDim = torch.IntTensor({3, br[2] - ul[2], br[1] - ul[1]})
      ht = imHeight
      wd = imWidth

      local newX = torch.Tensor({math.max(1, -ul[1]+1), math.min(br[1], wd) - ul[1]})
      local newY = torch.Tensor({math.max(1, -ul[2]+1), math.min(br[2], ht) - ul[2]})
      local oldX = torch.Tensor({math.max(1, ul[1]+1), math.min(br[1], wd)})
      local oldY = torch.Tensor({math.max(1, ul[2]+1), math.min(br[2], ht)})


      if rot ~= 0 then
          error('Currently we only support rotation == 0.')
      end

      -- mapping
      local newHm = torch.zeros(hm:size(1), ht, wd)
      hm = image.scale(hm:float(), newDim[3], newDim[2])
      newHm:sub(1, hm:size(1), oldY[1],oldY[2],oldX[1],oldX[2]):copy(hm:sub(1,hm:size(1),newY[1],newY[2],newX[1],newX[2]))

      -- Display heatmaps
      if false then
          local colorHms = {}
          for i = 1,16 do 
              colorHms[i] = colorHM(newHm[i])
              colorHms[i]:mul(.7):add(inp)
              w = image.display{image=colorHms[i],win=w}
              sys.sleep(2)
          end
      end

      return newHm
   end
   -------------------------------------------------------------------------------

   assert(self.opt.batchSize == 1, 'Multi-scale testing only support batchSize=1')

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local lossSum, accSum = 0.0, 0.0
   local N = 0
   local preds = torch.Tensor(dataloader:size(), self.opt.nClasses, 3)

   self.model:evaluate()
   for n, sample in dataloader:run(false, scales) do
      xlua.progress(n, size)
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)
      local batchSize = self.input:size(1)

      -- Output of original image
      local output = self.model:forward(self.input)
      local loss = self.criterion:forward(self.model.output, self.target)

      -- Output of flipped image
      output = applyFn(function (x) return x:clone() end, output)
      local flippedOut = self.model:forward(flip(self.input))
      flippedOut = applyFn(function (x) return flip(shuffleLR(x, self.opt)) end, flippedOut)
      output = applyFn(function (x,y) return x:add(y):div(2) end, output, flippedOut)

      -- Fuse heatmaps and map to original image resolution
      local imWidth, imHeight = sample.width[1], sample.height[1]
      local finalPredsHms = output[#output]
      local fuseHm = torch.zeros(self.opt.nClasses, imHeight, imWidth)
      for pyra = 1, #scales do
        local hm_img = getHeatmaps(imHeight, imWidth, sample.center[pyra], sample.scale[pyra], 0, 256, finalPredsHms[pyra])
        fuseHm = fuseHm + hm_img
      end
      fuseHm = fuseHm/#scales

      -- Get predictions
      local curImgIdx = sample.index[1]
      for p = 1,16 do
          local maxy, iy = fuseHm[p]:max(2)
          local maxv, ix = maxy:max(1)
          ix = torch.squeeze(ix)

          preds[curImgIdx][p][2] = ix
          preds[curImgIdx][p][1] = iy[ix]
          preds[curImgIdx][p][3] = maxy[ix]
      end  

      -- Visualize heatmaps
      if self.isDebug then
         local image = require('image')
         local im = image.load(sample.imgPath)
         local outIm = visualizer.drawOutput(im, fuseHm)
         win2=image.display{image=outIm, win=win2, legend=('Test output: %d | %d'):format(n, size)}
         sys.sleep(0)
      end

      -- Compute accuracies
      local acc, preds_hm = self:computeScore(output, sample.target)
      lossSum = lossSum + loss*batchSize
      accSum = accSum + acc*batchSize
      N = N + batchSize
      

      print((' Testing | Epoch: [%d][%d/%d]    Time %.3f  Data %.6f    Loss %.6f    Acc %.6f'):format(
       epoch, n, size, timer:time().real, dataTime, loss, acc))

      if epoch ~= 0 then
         self.testLogger:add{epoch, n, acc, loss}
      end

      timer:reset()
      dataTimer:reset()
   end
   self.model:training()

   -- -- Saving predictions
   local predFilePath = paths.concat(self.opt.save, self.opt.expID, 'pred_multiscale_' .. epoch .. '.h5')
   local predFile = hdf5.open(predFilePath, 'w')
   predFile:write('preds', preds)
   predFile:close()

   return accSum / N, lossSum / N
end

function Trainer:computeScore(output, target)
   -------------------------------------------------------------------------------
   -- Helpful functions for evaluation
   -------------------------------------------------------------------------------
   local function calcDists(preds, label, normalize)
       local dists = torch.Tensor(preds:size(2), preds:size(1))
       local diff = torch.Tensor(2)
       for i = 1,preds:size(1) do
           for j = 1,preds:size(2) do
               if label[i][j][1] > 1 and label[i][j][2] > 1 then
                   dists[j][i] = torch.dist(label[i][j],preds[i][j])/normalize[i]
               else
                   dists[j][i] = -1
               end
           end
       end
       return dists
   end

   local function getPreds(hm)
       assert(hm:dim() == 4, 'Input must be 4-D tensor')
       local max, idx = torch.max(hm:view(hm:size(1), hm:size(2), hm:size(3) * hm:size(4)), 3)
       local preds = torch.repeatTensor(idx, 1, 1, 2):float()
       preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hm:size(4) + 1 end)
       preds[{{}, {}, 2}]:add(-1):div(hm:size(3)):floor():add(1)
       return preds
   end

   local function distAccuracy(dists, thr)
       -- Return percentage below threshold while ignoring values with a -1
       if not thr then thr = .5 end
       if torch.ne(dists,-1):sum() > 0 then
           return dists:le(thr):eq(dists:ne(-1)):sum() / dists:ne(-1):sum()
       else
           return -1
       end
   end

   local function heatmapAccuracy(output, label, idxs, outputRes)
      -- Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
      -- First value to be returned is average accuracy across 'idxs', followed by individual accuracies
      local preds = getPreds(output)
      local gt = getPreds(label)
      local dists = calcDists(preds, gt, torch.ones(preds:size(1))*outputRes/10)
      local acc = {}
      local avgAcc = 0.0
      local badIdxCount = 0

      if not idxs then
        for i = 1,dists:size(1) do
            acc[i+1] = distAccuracy(dists[i])
            if acc[i+1] >= 0 then avgAcc = avgAcc + acc[i+1]
            else badIdxCount = badIdxCount + 1 end
        end
        acc[1] = avgAcc / (dists:size(1) - badIdxCount)
      else
        for i = 1,#idxs do
            acc[i+1] = distAccuracy(dists[idxs[i]])
            if acc[i+1] >= 0 then avgAcc = avgAcc + acc[i+1]
            else badIdxCount = badIdxCount + 1 end
        end
        acc[1] = avgAcc / (#idxs - badIdxCount)
      end
      return unpack(acc), preds
   end

  local jntIdxs = {mpii={1,2,3,4,5,6,11,12,15,16},flic={3,4,5,6,7,8,9,10,11}}

   if torch.type(target) == 'table' then
      return heatmapAccuracy(output[#output], target[#output], jntIdxs[self.opt.dataset], self.opt.outputRes)
   else
      return heatmapAccuracy(output, target, jntIdxs[self.opt.dataset], self.opt.outputRes)
   end
end

function Trainer:mapPreds(ratio, sample, preds_hm)
   local preds = preds_hm:clone():mul(ratio)
   local n = preds:size(1)
   local np = preds:size(2)
   local scale = sample.scale
   local offset = sample.offset

   for i = 1, n do
      -- offset
      preds[{i, {}, 1 }]:add(offset[i][1])
      preds[{i, {}, 2 }]:add(offset[i][2])
      -- scale      
      preds[i]:div(scale[i])
   end
   return preds
end

function Trainer:copyInputs(sample)
   self.input = sample.input:cuda()
   self.target = sample.target
   if torch.type(self.target) == 'table' then
      for s = 1, #self.target do self.target[s] = self.target[s]:cuda() end
   else
      self.target = self.target:cuda()
   end
end

-- function Trainer:learningRate(epoch)
--    -- Training schedule
--    local decay = 0
--    if string.find(self.opt.dataset, 'mpii') ~= nil then
--       decay = epoch >= 200 and 3 or epoch >= 170 and 2 or epoch >= 150 and 1 or 0
--    end
--    return self.opt.LR * math.pow(0.1, decay)
-- end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   for i = 1, #self.opt.schedule do
      if epoch >= self.opt.schedule[i] then 
         decay = i
      end
   end
   return self.opt.LR * math.pow(self.opt.gamma, decay)
end

return M.Trainer
