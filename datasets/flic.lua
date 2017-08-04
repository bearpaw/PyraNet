--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  CIFAR-10 dataset loader
--
--  Dataloader for FLIC

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/posetransforms_hg'

-------------------------------------------------------------------------------
-- Create dataset Class
-------------------------------------------------------------------------------

local M = {}
local FlicDataset = torch.class('resnet.FlicDataset', M)

function FlicDataset:__init(imageInfo, opt, split)
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
  self.meanstd = torch.load('gen/flic/meanstd.t7')
  self.minusMean = opt.minusMean
  self.opt = opt
end

function FlicDataset:get(i)
   local img = image.load(paths.concat('gen/flic/images', self.imageInfo.data['images'][i]))

   -- Generate samples
   local pts = self.imageInfo.labels['part'][i]
   local c = self.imageInfo.labels['center'][i]
   local s = self.imageInfo.labels['scale'][i]

   -- For single-person pose estimation with a centered/scaled figure
   local nParts = pts:size(1)
   local inp = t.crop(img, c, s, 0, self.inputRes)
   local out = torch.zeros(nParts, self.outputRes, self.outputRes)
   for i = 1,nParts do
     if pts[i][1] > 0 then -- Checks that there is a ground truth annotation
         t.drawGaussian(out[i], t.transform(torch.add(pts[i],1), c, s, 0, self.outputRes), 1)
     end
   end

   -- Data augmentation
   inp, out = self.augmentation(self, inp, out)

   -- nStack of target
   local target = out
   if self.nStack > 1 then
      target = {}
      for i = 1,self.nStack do target[i] = out end
   end
   
   return {
      input = inp,
      target = target,
      center = c,
      scale = s,
   }
end

function FlicDataset:getScale(i, scaleFactor)
   local scaleFactor = scaleFactor or 1
   local img = image.load(paths.concat('gen/flic/images', self.imageInfo.data['images'][i]))

   -- Generate samples
   local pts = self.imageInfo.labels['part'][i]
   local c = self.imageInfo.labels['center'][i]
   local s = self.imageInfo.labels['scale'][i]*scaleFactor

   -- For single-person pose estimation with a centered/scaled figure
   local nParts = pts:size(1)
   local inp = t.crop(img, c, s, 0, self.inputRes)
   local out = torch.zeros(nParts, self.outputRes, self.outputRes)
   for i = 1,nParts do
     if pts[i][1] > 0 then -- Checks that there is a ground truth annotation
         t.drawGaussian(out[i], t.transform(torch.add(pts[i],1), c, s, 0, self.outputRes), 1)
     end
   end

   -- Data augmentation
   inp, out = self.augmentation(self, inp, out)

   -- nStack of target
   local target = out
   if self.nStack > 1 then
      target = {}
      for i = 1,self.nStack do target[i] = out end
   end
   
   return {
      input = inp,
      target = target,
      center = c,
      scale = s,
      imname = self.imageInfo.data['images'][i],
   }
end

function FlicDataset:size()
   return self.imageInfo.labels.nsamples - (self.imageInfo.labels.nsamples%self.opt.nGPU)
end

function FlicDataset:preprocess()
   return function(img)
      return img
   end
end

function FlicDataset:augmentation(input, label)
  if input:max() > 2 then
     input:div(255)
  end
  -- Augment data (during training only)
  if self.split ~= 'val' then
      local s = torch.randn(1):mul(self.scaleFactor):add(1):clamp(1-self.scaleFactor,1+self.scaleFactor)[1]
      local r = torch.randn(1):mul(self.rotFactor):clamp(-2*self.rotFactor,2*self.rotFactor)[1]

      -- Color
      input[{1, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)
      input[{2, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)
      input[{3, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)

      -- Scale/rotation
      if torch.uniform() <= .6 then r = 0 end
      local inp,out = self.inputRes, self.outputRes
      input = t.crop(input, {(inp+1)/2,(inp+1)/2}, inp*s/200, r, inp)
      label = t.crop(label, {(out+1)/2,(out+1)/2}, out*s/200, r, out)

      -- Flip
      if torch.uniform() <= .5 then
          input = t.flip(input)
          label = t.flip(t.shuffleLR(label, self.dataset))
      end
  end

  -- Image mean
  if self.minusMean  then 
    input = t.colorNormalize(input, self.meanstd) 
  end

  return input, label
end

return M.FlicDataset
