--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt)
   -- The train and val loader
   local loaders = {}

   sets = {'train', 'val'}
   if opt.testRelease then sets = {'train', 'val', 'test'} end

   for i, split in ipairs(sets) do
      local dataset = datasets.create(opt, split)
      loaders[i] = M.DataLoader(dataset, opt, split)
   end

   return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
   local manualSeed = opt.manualSeed
   local function init()
      require('datasets/' .. opt.dataset)
   end
   local function main(idx)
      if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
      end
      torch.setnumthreads(1)
      _G.dataset = dataset
      _G.preprocess = dataset:preprocess()
      return dataset:size()
   end

   local threads, sizes = Threads(opt.nThreads, init, main)
   self.threads = threads
   self.__size = sizes[1][1]
   self.batchSize = opt.batchSize
   self.inputRes = opt.inputRes
   self.outputRes = opt.outputRes
   self.nStack = opt.nStack
end

function DataLoader:size()
   return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run(randPerm, scaleFactor)
   local randPerm = true
   if randPerm_ ~= nil then
      randPerm = randPerm_
   end
   local threads = self.threads
   local size, batchSize = self.__size, self.batchSize
   local perm = torch.randperm(size)
   if not randPerm then
      perm = torch.range(1, size)
   end

   local idx, sample = 1, nil
   local nStack = self.nStack
   local scales = scaleFactor or {1}

   local function enqueue()
      while idx <= size and threads:acceptsjob() do
         local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
         threads:addjob(
            function(indices)
               local sz = #scales
               local batch, target = nil, nil
               local scale, offset, center, index, width, height = {}, {}, {}, {}, {}, {}-- for testing pose 
               local idx = indices[1]
               local imgPath
               
               for i, scaleFactor in ipairs(scales) do
                  local sample = _G.dataset:get(idx, scaleFactor)
                  local input = _G.preprocess(sample.input)

                  if not batch then 
                     batch = input:view(1,unpack(input:size():totable())) 
                  else 
                     batch = batch:cat(input:view(1,unpack(input:size():totable())),1) 
                  end

                  if not target then 
                     target = sample.target:view(1,unpack(sample.target:size():totable())) 
                  else 
                     target = target:cat(sample.target:view(1,unpack(sample.target:size():totable())),1) 
                  end

                  -- for testing pose
                  scale[i] = sample.scale
                  offset[i] = sample.offset
                  center[i] = sample.center
                  index[i] = idx
                  width[i] = sample.width
                  height[i] = sample.height
                  imgPath = sample.imgPath
                  -- label[i] = sample.label or -1
               end

               -- Set up label for intermediate supervision
               if nStack > 1 then
                  local targetTable = {}
                  for s = 1, nStack do table.insert(targetTable, target) end
                  target = targetTable
               end
               collectgarbage()
               return {
                  input = batch,
                  target = target,
                  scale = scale,  -- for testing pose
                  offset = offset,  -- for testing pose
                  center = center, -- for testing pose
                  index = index,
                  width = width,
                  height = height,
                  imgPath = imgPath,
                  -- label = label,
               }
            end,
            function(_sample_)
               sample = _sample_
            end,
            indices
         )
         idx = idx + batchSize
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      n = n + 1
      return n, sample
   end

   return loop
end

return M.DataLoader
