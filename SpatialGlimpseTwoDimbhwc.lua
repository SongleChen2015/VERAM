------------------------------------------------------------------------
--[[ SpatialGlimpseTwoDimbhwc ]]--

------------------------------------------------------------------------
local SpatialGlimpseTwoDimbhwc, parent = torch.class("SpatialGlimpseTwoDimbhwc", "nn.Module") 
 
function SpatialGlimpseTwoDimbhwc:__init(size, depth, scale)
   require 'nnx'
   --print(torch.type(size))
   if torch.type(size)=='table' then
      self.height = size[1]
      self.width = size[2]
   else
      self.width = size
      self.height = size
   end
   self.depth = depth or 3
   self.scale = scale or 2
   --print(torch.type(self.width))
   assert(torch.type(self.width) == 'number')
   assert(torch.type(self.height) == 'number')
   assert(torch.type(self.depth) == 'number')
   assert(torch.type(self.scale) == 'number') 
   parent.__init(self)
   self.gradInput = {torch.Tensor(), torch.Tensor()}
   if self.scale == 2 then
      self.module = nn.SpatialAveragePooling(2,2,2,2)
   else
      self.module = nn.SpatialReSampling{oheight=self.height,owidth=self.width}
   end
   self.modules = {self.module}
end

function SpatialGlimpseTwoDimbhwc:updateOutput(inputTable)
   assert(torch.type(inputTable) == 'table')
   assert(#inputTable >= 2)
   local input, location = unpack(inputTable)
   input, location = self:toBatch(input, 3), self:toBatch(location, 1)
   assert(input:dim() == 4 and location:dim() == 2)
   
   --bchw
   self.output:resize(input:size(1), self.height, self.width, input:size(4))
   outcoord = torch.clamp(torch.round(location*12), 1, 12)
   
   for sampleIdx=1,self.output:size(1) do
      local outputSample = self.output[sampleIdx]
      local inputSample = input[sampleIdx]
        y = outcoord[sampleIdx][1]
        x = outcoord[sampleIdx][2]
         --input is bhwc
        outputSample:copy(inputSample:narrow(1, y , self.height):narrow(2, x, self.width))
   end
   self.output = self:fromBatch(self.output, 1)

   return self.output
end

function SpatialGlimpseTwoDimbhwc:updateGradInput(inputTable, gradOutput)
   local input, location = unpack(inputTable)
   if #self.gradInput ~= 2 then
      self.gradInput = {input.new(), input.new()}
   end
   local gradInput, gradLocation = unpack(self.gradInput)
   input, location = self:toBatch(input, 3), self:toBatch(location, 1)
   gradOutput = self:toBatch(gradOutput, 3)
   
   gradInput:resizeAs(input):zero()
   gradLocation:resizeAs(location):zero() -- no backprop through location
   
   gradOutput = gradOutput:view(input:size(1), self.height, self.width, input:size(4))
   
   outcoord = torch.clamp(torch.round(location*12), 1, 12)
   
   for sampleIdx=1,gradOutput:size(1) do
      local gradOutputSample = gradOutput[sampleIdx]
      local gradInputSample = gradInput[sampleIdx]
     
      local inputSample = input[sampleIdx]
      y = outcoord[sampleIdx][1]
      x = outcoord[sampleIdx][2]
      local pad = gradInputSample:narrow(1, y, self.height):narrow(2, x, self.width)
         
      pad:copy(gradOutputSample) 
   end
   
   self.gradInput[1] = self:fromBatch(gradInput, 1)
   self.gradInput[2] = self:fromBatch(gradLocation, 1)
   
   return self.gradInput
end
