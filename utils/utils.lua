local Transformer = require 'datasets.posetransforms'
local utils = {}


function utils.narrowTable(startIdx, endIdx, inputTable)
  assert(startIdx >= 1 and startIdx <= #inputTable, 'startIdx should > 1 and < #inputTable')
  assert(endIdx >= startIdx and endIdx <= #inputTable, 'endIdx > startIdx and endIdx <= #inputTable')
  local outputTable = {}
  for i = startIdx, endIdx do
    table.insert(outputTable, inputTable[i])
  end
  return outputTable
end


function utils.list_nngraph_modules(g)
----------------------------------------------------------
-- List modules given an nngraph g
----------------------------------------------------------
  local omg = {}
  for i,node in ipairs(g.forwardnodes) do
      local m = node.data.module
      if m then
        table.insert(omg, m)
      end
   end
   return omg
end


function utils.list_nnsequencer_modules(seq)
----------------------------------------------------------
-- List modules given a Sequencer
----------------------------------------------------------
   return utils.listModules(seq:get(1):get(1))
end


function utils.listModules(net)
----------------------------------------------------------
-- List modules given an nn model
----------------------------------------------------------
  local t = torch.type(net)
  local moduleList
  if t == 'nn.gModule' then
    moduleList = utils.list_nngraph_modules(net)
  elseif t == 'nn.Sequencer' then
    moduleList = utils.list_nnsequencer_modules(net)    
  else
    moduleList = net:listModules()
  end
  return moduleList
end


-------------------------------------------------------------------------------
-- Flipping functions
-------------------------------------------------------------------------------
function utils.shuffleLR(x, opt)
  local dim
  if x:nDimension() == 4 then
      dim = 2
  else
      assert(x:nDimension() == 3)
      dim = 1
  end

  local matchedParts
  if string.find(opt.dataset, 'mpii') then
      matchedParts = {
          {1,6},   {2,5},   {3,4},
          {11,16}, {12,15}, {13,14}
      }
  elseif opt.dataset == 'flic' then
      -- matched_parts = {
      --     {1,4}, {2,5}, {3,6}, {7,8}, {9,10}
      -- }      
      matchedParts = {
          {1,2}, {4,8}, {5,9}, {6,10}, {7,11}
      }
  elseif opt.dataset == 'lsp' then
      matchedParts = {
          {1,6}, {2,5}, {3,4}, {7,12}, {8,11}, {9,10}
      }
  elseif opt.dataset == 'coco' then
    matchedParts = {
        {2,3},   {4,5},   {6,7}, {8,9}, {10,11},
        {12,13}, {14,15}, {16,17}
    }
  elseif opt.dataset == 'cooking' then
      matchedParts = {
          {3,4}, {5,6}, {7,8}, {9,10}
      }
  elseif opt.dataset == 'youtube' or opt.dataset == 'youtube-single' 
      or string.find(opt.dataset, 'bbcpose') ~= nil 
      or string.find(opt.dataset, 'videopose2') ~= nil then -- == 'bbcpose-single' or dataset == 'bbcpose' then        
      matchedParts = {
          {2,3}, {4,5}, {6,7}
      }
  end

  for i = 1,#matchedParts do
      local idx1, idx2 = unpack(matchedParts[i])
      local tmp = x:narrow(dim, idx1, 1):clone()
      x:narrow(dim, idx1, 1):copy(x:narrow(dim, idx2, 1))
      x:narrow(dim, idx2, 1):copy(tmp)
  end

  return x
end


function utils.flip(x)
    require 'image'
    local y = torch.FloatTensor(x:size())
    for i = 1, x:size(1) do
        image.hflip(y[i], x[i]:float())
    end
    return y:typeAs(x)
end


local function recursiveApplyFn(fn, t, t2)
    -- Helper function for applying an operation whether passed a table or tensor
    local t_ = {}
    if type(t) == "table" then
        if t2 then
            for i = 1,#t do t_[i] = recursiveApplyFn(fn, t[i], t2[i]) end
        else
            for i = 1,#t do t_[i] = recursiveApplyFn(fn, t[i]) end
        end
    else t_ = fn(t, t2) end
    return t_
end

function utils.applyFn(fn, t, t2)
    return recursiveApplyFn(fn, t, t2)
end


local function postprocess(heatmap, p)
   assert(heatmap:size(1) == p:size(1))
   local scores = torch.zeros(p:size(1),p:size(2),1)

   -- Very simple post-processing step to improve performance at tight PCK thresholds
   for i = 1,p:size(1) do
      for j = 1,p:size(2) do
         local hm = heatmap[i][j]
         local pX,pY = p[i][j][1], p[i][j][2]
         scores[i][j] = hm[pY][pX]
         if pX > 1 and pX < hm:size(2) and pY > 1 and pY < hm:size(1) then
            local diff = torch.Tensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})
            p[i][j]:add(diff:sign():mul(.25))
         end
      end
   end
   return p:add(0.5)
end


function utils.finalPreds(heatmaps, centers, scales)
    local n = heatmaps:size(1)
    local np = heatmaps:size(2)

    local preds = torch.Tensor(n, np, 2)
    local preds_tf = torch.Tensor(n, np, 3) -- save score also

    for sidx = 1, heatmaps:size(1) do
      local hms = heatmaps[sidx]
      local center = centers[sidx]
      local scale = scales[sidx]

      if hms:size():size() == 3 then hms = hms:view(1, hms:size(1), hms:size(2), hms:size(3)) end

      -- Get locations of maximum activations
      local max, idx = torch.max(hms:view(hms:size(1), hms:size(2), hms:size(3) * hms:size(4)), 3)
      local pred = torch.repeatTensor(idx, 1, 1, 2):float()
      pred[{{}, {}, 1}]:apply(function(x) return (x - 1) % hms:size(4) + 1 end)
      pred[{{}, {}, 2}]:add(-1):div(hms:size(3)):floor():add(1)
      local predMask = max:gt(0):repeatTensor(1, 1, 2):float()
      pred:add(-1):cmul(predMask):add(1)

      pred = postprocess(hms, pred)
      -- Get transformed coordinates
      local pred_tf = torch.zeros(pred:size())
      for i = 1,hms:size(1) do        -- Number of samples
          for j = 1,hms:size(2) do    -- Number of output heatmaps for one sample
              pred_tf[i][j] = Transformer.transform(pred[i][j],center,scale,0,hms:size(3),true)
          end
      end
      pred_tf = torch.cat(pred_tf:float(), max:float(), 3)
      preds[sidx] = pred
      preds_tf[sidx] = pred_tf
    end

    return preds, preds_tf
end


function utils.OffsetFieldsfinalPreds(heatmaps, centers, scales)
   local n = heatmaps:size(1)
   local np = heatmaps:size(2)/2
   local height, width = heatmaps:size(3), heatmaps:size(4)

   local preds = torch.Tensor(n, np, 2)
   local preds_tf = torch.Tensor(n, np, 2)

   -- mapping
   assert(height == width)
   heatmaps:mul(height):floor():add(1)

   for sidx = 1, heatmaps:size(1) do
      local offset = heatmaps[sidx]
      local center = centers[sidx]
      local scale = scales[sidx]

      local hms = torch.zeros(np, height, width);
      for p = 1, np do
         local hmsX = offset[(p-1)*2+1]
         local hmsY = offset[p*2]

         for h = 1, height do
            for w = 1, width do
               local dx, dy = w - hmsX[h][w], h - hmsY[h][w]
               if dx > 0 and dy > 0 and dx <= width and dy <= height then
                  hms[p][dy][dx] = hms[p][dy][dx] + 1
               end
            end
         end
      end

      -- Get locations of maximum activations
      hms = hms:view(1, hms:size(1), hms:size(2), hms:size(3))
      local max, idx = torch.max(hms:view(hms:size(1), hms:size(2), hms:size(3) * hms:size(4)), 3)
      local pred = torch.repeatTensor(idx, 1, 1, 2):float()
      pred[{{}, {}, 1}]:apply(function(x) return (x - 1) % hms:size(4) + 1 end)
      pred[{{}, {}, 2}]:add(-1):div(hms:size(3)):floor():add(1)
      local predMask = max:gt(0):repeatTensor(1, 1, 2):float()
      pred:add(-1):cmul(predMask):add(1)

      pred = postprocess(hms, pred)
      -- Get transformed coordinates
      local pred_tf = torch.zeros(pred:size())
      for i = 1,hms:size(1) do        -- Number of samples
          for j = 1,hms:size(2) do    -- Number of output heatmaps for one sample
              pred_tf[i][j] = Transformer.transform(pred[i][j],center,scale,0,hms:size(3),true)
          end
      end
      preds[sidx] = pred
      preds_tf[sidx] = pred_tf
   end
   print(preds, preds_tf)
   
   return preds, preds_tf
end


return utils