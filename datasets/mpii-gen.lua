--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Copyright (c) 2016, YANG Wei
--  Script to prepare MPII dataset
--  The annotations are adopted from: https://github.com/anewell/pose-hg-train/tree/master/data/mpii/annot

local hdf5 = require 'hdf5'

local M = {}

local function convertMPII(file, namelist)
   local data, labels = {}, {}
   local a = hdf5.open(file, 'r')
   local namesFile = io.open(namelist, 'r')


   -- Read in annotation information
   local tags = {'part', 'center', 'scale', 'normalize', 'torsoangle', 'visible'}
   for _,tag in ipairs(tags) do 
      labels[tag] = a:read(tag):all() 
   end
   labels['nsamples'] = labels['part']:size()[1]

   -- Load in image file names (reading strings wasn't working from hdf5)
   data['images'] = {}
   local toIdxs = {}
   local idx = 1
   for line in namesFile:lines() do
     data['images'][idx] = line
     if not toIdxs[line] then toIdxs[line] = {} end
     table.insert(toIdxs[line], idx)
     idx = idx + 1
   end
   namesFile:close()

   -- This allows us to reference multiple people who are in the same image
   data['imageToIdxs'] = toIdxs

   return {
      data = data,
      labels = labels,
   }
end

local function computeMean( train )
   print('==> Compute image mean and std')
   local meanstd
   local data, labels = train.data, train.labels
   if not paths.dirp('gen/mpii') then
      paths.mkdir('gen/mpii')
   end   
   if not paths.filep('gen/mpii/meanstd.t7') then
      local size_ = labels['nsamples']
      local rgbs = torch.Tensor(3, size_)

      for idx = 1, size_ do
         xlua.progress(idx, size_)
         local imgpath = paths.concat('data/mpii/images', data['images'][idx])
         local imdata = image.load(imgpath)
         rgbs:select(2, idx):copy(imdata:view(3, -1):mean(2))
      end

      local mean = rgbs:mean(2):squeeze()
      local std = rgbs:std(2):squeeze()

      meanstd = {
         mean = mean,
         std = std
      }

      torch.save('gen/mpii/meanstd.t7', meanstd)
   else
      meanstd = torch.load('gen/mpii/meanstd.t7')
   end

   print(('    mean: %.4f %.4f %.4f'):format(meanstd.mean[1], meanstd.mean[2], meanstd.mean[3]))
   print(('     std: %.4f %.4f %.4f'):format(meanstd.std[1], meanstd.std[2], meanstd.std[3]))
   return meanstd
end

function M.exec(opt, cacheFile)
   local trainData = convertMPII('data/mpii/train.h5', 'data/mpii/train_images.txt')
   local validData = convertMPII('data/mpii/valid.h5', 'data/mpii/valid_images.txt')
   local testData = convertMPII('data/mpii/test.h5', 'data/mpii/test_images.txt')

   -- Compute image mean
   computeMean(trainData)

   print(" | saving MPII dataset to " .. cacheFile)
   torch.save(cacheFile, {
      train = trainData,
      val = validData,
      test = testData,
   })
   print("  train: ".. trainData.labels.nsamples)
   print("  valid: ".. validData.labels.nsamples)
   print("  test: ".. testData.labels.nsamples)
end

return M
