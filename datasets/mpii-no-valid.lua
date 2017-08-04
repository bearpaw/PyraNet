--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  MPII dataset loader (from  (Newell))
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/posetransforms'


-------------------------------------------------------------------------------
-- Helper Functions
-------------------------------------------------------------------------------
local getTransform = t.getTransform
local transform = t.transform
local crop = t.crop2
local drawGaussian = t.drawGaussian
local shuffleLR = t.shuffleLR
local flip = t.flip
local colorNormalize = t.colorNormalize

-------------------------------------------------------------------------------
-- Create dataset Class
-------------------------------------------------------------------------------

local M = {}
local MpiiDataset = torch.class('resnet.MpiiDataset', M)

function MpiiDataset:__init(imageInfo, opt, split)
  assert(imageInfo[split], split)
  self.imageInfo = imageInfo[split]
  self.split = split
  -- Some arguments
  self.inputRes = opt.inputRes
  self.outputRes = opt.outputRes
  -- Options for augmentation
  self.scaleFactor = opt.scaleFactor
  self.rotFactor = opt.rotFactor
  self.dataset = opt.dataset
  self.nStack = opt.nStack
  self.meanstd = torch.load('gen/mpii/meanstd.t7')
  self.nGPU = opt.nGPU
  self.batchSize = opt.batchSize
  self.minusMean = opt.minusMean
  self.gsize = opt.gsize
  self.bg = opt.bg
  self.rotProbab = opt.rotProbab
end

function MpiiDataset:get(i, scaleFactor)
   local scaleFactor = scaleFactor or 1
   local img = image.load(paths.concat('data/mpii/images', self.imageInfo.data['images'][i]))

   -- Generate samples
   local pts = self.imageInfo.labels['part'][i]
   local c = self.imageInfo.labels['center'][i]
   local s = self.imageInfo.labels['scale'][i]*scaleFactor

   -- For single-person pose estimation with a centered/scaled figure
   local nParts = pts:size(1)
   local inp = crop(img, c, s, 0, self.inputRes)
   local out = self.bg == 'true' and torch.zeros(nParts+1, self.outputRes, self.outputRes) 
                                or torch.zeros(nParts, self.outputRes, self.outputRes)
   for i = 1,nParts do
     if pts[i][1] > 0 then -- Checks that there is a ground truth annotation
         drawGaussian(out[i], transform(torch.add(pts[i],1), c, s, 0, self.outputRes), self.gsize)
     end
   end

   if self.bg == 'true' then
      out[nParts+1], _ = torch.max(out:sub(1, nParts, 1, self.outputRes, 1, self.outputRes), 1)
   end

   -- Data augmentation
   inp, out = self.augmentation(self, inp, out)
   collectgarbage()
   return {
      input = inp,
      target = out,
      center = c,
      scale = s,
      width = img:size(3),
      height = img:size(2),
      imgPath = paths.concat('data/mpii/images', self.imageInfo.data['images'][i])
   }
end

function MpiiDataset:size()
  if self.split == 'test' then
    return self.imageInfo.labels.nsamples
  end
  
   local nSamples = self.imageInfo.labels.nsamples - (self.imageInfo.labels.nsamples%self.nGPU)
   nSamples = nSamples - nSamples%self.batchSize

   return nSamples
end

function MpiiDataset:preprocess()
   return function(img)
      if img:max() > 2 then
         img:div(255)
      end
      return self.minusMean == 'true' and colorNormalize(img, self.meanstd) or img
   end
end

function MpiiDataset:augmentation(input, label)
  -- Augment data (during training only)
  if self.split == 'train' then
      local s = torch.randn(1):mul(self.scaleFactor):add(1):clamp(1-self.scaleFactor,1+self.scaleFactor)[1]
      local r = torch.randn(1):mul(self.rotFactor):clamp(-2*self.rotFactor,2*self.rotFactor)[1]

      -- Color
      input[{1, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)
      input[{2, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)
      input[{3, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)

      -- Scale/rotation
      if torch.uniform() <= (1-self.rotProbab) then r = 0 end
      local inp,out = self.inputRes, self.outputRes
      input = crop(input, torch.Tensor({(inp+1)/2,(inp+1)/2}), inp*s/200, r, inp)
      label = crop(label, torch.Tensor({(out+1)/2,(out+1)/2}), out*s/200, r, out)

      -- Flip
      if torch.uniform() <= .5 then
          input = flip(input)
          label = flip(shuffleLR(label, self.dataset))
      end
  end

  return input, label
end

return M.MpiiDataset