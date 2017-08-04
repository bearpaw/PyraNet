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

local function merge(db)
   local n = #db
   local data, labels = {}, {}

   -- Merge labels
   local tags = {'part', 'center', 'scale', 'normalize', 'torsoangle', 'visible'}
   for _,tag in ipairs(tags) do 
      local tb = {}
      for i = 1, n do
         table.insert(tb, db[i].labels[tag])
      end
      labels[tag] = torch.cat(tb, 1)
   end
   labels['nsamples'] = labels['part']:size()[1]

   -- Merge data
   data['images'] = {}
   for i = 1, n do 
      for j = 1, #db[i].data['images'] do
         table.insert(data['images'], db[i].data['images'][j])
      end
   end

   -- print(db[1].data['imageToIdxs'])

   local toIdxs = {}
   for i = 1, n do 
      local imageToIdxs = db[i].data['imageToIdxs']
      for key, val in pairs(imageToIdxs) do
         if not toIdxs[key] then 
            toIdxs[key] = val
         else
            toIdxs[key] = table.concat(toIdxs[key], val)
         end
      end
   end
   data['imageToIdxs'] = toIdxs

   return {
      data = data,
      labels = labels,
   }

end

function M.exec(opt, cacheFile)
   local trainData = convertMPII('data/mpii/train.h5', 'data/mpii/train_images.txt')
   local validData = convertMPII('data/mpii/valid.h5', 'data/mpii/valid_images.txt')
   local testData = convertMPII('data/mpii/test.h5', 'data/mpii/test_images.txt')

   local fullTrainData = merge({trainData, validData})

   print(" | saving MPII dataset to " .. cacheFile)
   torch.save(cacheFile, {
      train = fullTrainData,
      val = validData,
      test = testData,
   })
   print("  train: ".. fullTrainData.labels.nsamples)
   print("  valid: ".. validData.labels.nsamples)
   print("  test: ".. testData.labels.nsamples)
   -- print("  train+val: ".. trainData.labels.nsamples+validData.labels.nsamples)
   -- print("  fullTrain: ".. fullTrainData.labels.nsamples)
   -- os.exit()
end

return M
