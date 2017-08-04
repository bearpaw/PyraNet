require 'xlua'

local M = {}
local Plotter = torch.class('Plotter', M)


--------------------------------------------------------------------------------
--  Helper Functions
--------------------------------------------------------------------------------
local function loadFile(filename, title)
  local title = title or 'default'
	local f = assert(io.open(filename, "r"))
	local str = f:read("*l")
	local names = string.split(str, "\t")
  -- Leading and trailing whitespace removed
  -- https://www.rosettacode.org/wiki/Strip_whitespace_from_a_string/Top_and_tail#Lua
  for i = 1, #names do
    names[i]=names[i]:match( "^%s*(.-)%s*$" )   
  end                                             
	local lines = {}
	for _, name in pairs(names) do
		lines[name] = {}
	end
	-- read the lines in table 'lines'
  for line in f:lines() do
  	local nums = string.split(line, "\t")
  	for i, num in pairs(nums) do 
  		table.insert(lines[names[i]], num)
  	end
  end
  return {
  	title=title,
  	names=names,
  	lines=lines,
	}
end

local function plotsymbol(name,list)
   if #list > 1 then
      local nelts = #list
      local plot_y = torch.Tensor(nelts)
      for i = 1,nelts do
         plot_y[i] = list[i]
      end
      for _,style in ipairs(self.styles[name]) do
         table.insert(plots, {self.names[name], plot_y, style})
      end
      plotit = true
   end
end

local function contains(table, val)
   for i=1,#table do
      if table[i] == val then 
         return true
      end
   end
   return false
end

--------------------------------------------------------------------------------
--  Public Interfaces
--------------------------------------------------------------------------------
function Plotter:__init(fileNames, logscale)
	self.Logger = {}
  self.logscale = logscale or false
	if torch.type(fileNames) == 'string' then
		table.insert(self.Logger, loadFile(fileNames))
	elseif torch.type(fileNames) == 'table' then
		for title, fileName in pairs(fileNames) do
			-- table.insert(self.Logger, loadFile(fileName, title))
			self.Logger[title] = loadFile(fileName, title)
		end
	end

end

function Plotter:plot(fileNames, names, skipStep)
  local skipStep = skipStep or 0
	if not xlua.require('gnuplot') then
    if not self.warned then
       print('<Logger> warning: cannot plot with this version of Torch')
       self.warned = true
    end
    return
 	end

  local plots = {}
	local plotsymbol =
	  function(name,list)
	     if #list > 1 then
	     		local style = '-'
	        local nelts = #list
	        local plot_y = torch.Tensor(nelts-skipStep)
          local plot_x = torch.Tensor(nelts-skipStep)
	        for i = skipStep+1,nelts do
	           plot_y[i-skipStep] = list[i]
             plot_x[i-skipStep] = i
	        end
	        table.insert(plots, {name, plot_x, plot_y, style})
	     end
	  end

 	for _, fineName in pairs(fileNames) do
 		local Logger = self.Logger[fineName]
		for _, name in pairs(names) do
			if contains(Logger['names'], name) then
       	plotsymbol(Logger['title'] ..'-' .. name, Logger['lines'][name])
      else
        print(('<Plotter> warning: name [%s] does not exists.'):format(name))
        return
      end
    end
 	end

 	self.figure = gnuplot.figure(self.figure)
	if self.logscale then gnuplot.logscale('on') end
	gnuplot.plot(plots)
	if self.plotRawCmd then gnuplot.raw(self.plotRawCmd) end
	gnuplot.grid('on')
	-- gnuplot.title('<Logger::' .. self.name .. '>')
end

return M.Plotter
