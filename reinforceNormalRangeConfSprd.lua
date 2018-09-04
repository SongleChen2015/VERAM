------------------------------------------------------------------------
--[[ ReinforceNormalRangeConfSprd ]]-- 
-- Inputs are mean (mu) of multivariate normal distribution. 
-- Ouputs are samples drawn from these distributions.
------------------------------------------------------------------------
require 'nn'
local ReinforceNormalRangeConfSprd, parent = torch.class("ReinforceNormalRangeConfSprd", "nn.Reinforce")

function ReinforceNormalRangeConfSprd:__init(stdev, stochastic,lowbound, upbound)
   parent.__init(self, stochastic)
   self.stdev = stdev
   self.lowbound = lowbound
   self.upbound = upbound
   if not stdev then
      self.gradInput = {torch.Tensor(), torch.Tensor()}
   end
   if not self.lowbound then
     self.lowbound = 0
   end
   
   if not self.upbound then
     self.upbound = 1
   end
   
   --for modelnet10 and modelnet40 it is 12*12
   self.gridsize = 12

end

function ReinforceNormalRangeConfSprd:updateOutput(input)
   local mean, stdev = input[1], self.stdev
   
   --[[ comment, for now input[2] is view confidence
   if torch.type(input) == 'table' then
      -- input is {mean, stdev}
      assert(#input == 2)
      mean, stdev = unpack(input)
   end
   --]]
   assert(stdev)
   
   self.output:resizeAs(mean)
   if self.stochastic or self.train ~= false then
      self.output:normal()
      -- multiply by standard deviations
      if torch.type(stdev) == 'number' then
         self.output:mul(stdev)
      elseif torch.isTensor(stdev) then
         if stdev:dim() == mean:dim() then
            assert(stdev:isSameSizeAs(mean))
            self.output:cmul(stdev)
         else
            assert(stdev:dim()+1 == mean:dim())
            self._stdev = self._stdev or stdev.new()
            self._stdev:view(stdev,1,table.unpack(stdev:size():totable()))
            self.__stdev = self.__stdev or stdev.new()
            self.__stdev:expandAs(self._stdev, mean)
            self.output:cmul(self.__stdev)
         end
      else
         error"unsupported mean type"
      end
      
      -- re-center the means to the mean
      self.output:add(mean)
   else
      -- use maximum a posteriori (MAP) estimate
      self.output:copy(mean)
   end
   
   --for normalize, force into the ideal range
   self.output = torch.clamp(self.output, self.lowbound, self.upbound)
   
   return self.output
   
end

function ReinforceNormalRangeConfSprd:updateGradInput(input, gradOutput)
   -- Note that gradOutput is ignored
   -- f : normal probability density function
   -- x : the sampled values (self.output)
   -- u : mean (mu) (mean)
   -- s : standard deviation (sigma) (stdev)
   confidence = input[2]
   
   local mean, stdev = input[1], self.stdev
   local gradMean, gradStdev = self.gradInput, nil
   --[[ comment, for now input[2] is the view confidence
   if torch.type(input) == 'table' then
      mean, stdev = unpack(input)
      gradMean, gradStdev = unpack(self.gradInput)
   end
   --]]
   assert(stdev)   
    
   -- Derivative of log normal w.r.t. mean :
   -- d ln(f(x,u,s))   (x - u)
   -- -------------- = -------
   --      d u           s^2
   
   gradMean:resizeAs(mean)
   -- (x - u)
   gradMean:copy(self.output):add(-1, mean)
   
   -- divide by squared standard deviations
   if torch.type(stdev) == 'number' then
      gradMean:div(stdev^2)
   else
      if stdev:dim() == mean:dim() then
         gradMean:cdiv(stdev):cdiv(stdev)
      else
         gradMean:cdiv(self.__stdev):cdiv(self.__stdev)
      end
   end
   
   -- multiply by reward
   gradMean:cmul(self:rewardAs(mean) )
   -- multiply by -1 ( gradient descent on mean )
   gradMean:mul(-1)
   
   --for adjust the gradient
   outcoord = torch.clamp(torch.round(self.output*12), 1, 12)
  
   --get the batch size
   bsize = (#(input[1]))[1]
   cf = confidence:view(bsize,12,12)
    --prepare the first dim index,must be same with cf
   t1 = outcoord:narrow(2,1,1):expand(bsize,12):clone():view(bsize,1,12)
   --prepare the second dim index
   t2 = outcoord:narrow(2,2,1)
   --the first dim is batch , first gather alonge the first coord,then the second
   --expand it to both coordination
   confout = cf:gather(2,t1):view(bsize,12):gather(2,t2):expand(bsize, 2)
   
   local tempreward = self:rewardAs(mean)
   rewardlessmask = torch.lt(tempreward, 0)
   rewardlargemask = torch.gt(tempreward,0)
   
   if torch.any(rewardlessmask) then
     --check low bound
     maskmin = torch.eq(outcoord, 1) --frist frame
     if torch.any(maskmin)  then 
       masklessmin = torch.lt(input[1]-self.output, 0) --input is less than the first frame
       if torch.any(masklessmin) then
         maskmincombine = maskmin:cmul(masklessmin):cmul(rewardlessmask) --only affects on error class
         gradMean[maskmincombine] = gradMean[maskmincombine]*-1
       end
     end

     --check upper bound
     maskmax = torch.eq(outcoord, 12)
     if torch.any(maskmax) then
       makslargemax = torch.gt(input[1]-self.output, 0)
       if torch.any(makslargemax) then
         maskmaxcombine = maskmax:cmul(makslargemax):cmul(rewardlessmask)
         gradMean[maskmaxcombine] = gradMean[maskmaxcombine]*-1
       end
     end
   end
   
   --multiply confidence
   --class is error *(1-conf)
   gradMean[rewardlessmask] = gradMean[rewardlessmask]:cmul(confout[rewardlessmask]*-1+1)
   --class is right * conf
   gradMean[rewardlargemask] = gradMean[rewardlargemask]:cmul(confout[rewardlargemask])
   
   -- Derivative of log normal w.r.t. stdev :
   -- d ln(f(x,u,s))   (x - u)^2 - s^2
   -- -------------- = ---------------
   --      d s              s^3
   
   if gradStdev then
      gradStdev:resizeAs(stdev)
      -- (x - u)^2
      gradStdev:copy(self.output):add(-1, mean):pow(2)
      -- subtract s^2
      self._stdev2 = self._stdev2 or stdev.new()
      self._stdev2:resizeAs(stdev):copy(stdev):cmul(stdev)
      gradStdev:add(-1, self._stdev2)
      -- divide by s^3
      self._stdev2:cmul(stdev):add(0.00000001)
      gradStdev:cdiv(self._stdev2)
      -- multiply by reward
      gradStdev:cmul(self:rewardAs(stdev))
       -- multiply by -1 ( gradient descent on stdev )
      gradStdev:mul(-1)
   end
   
   gradConf = input[2].new()
   gradConf:resizeAs(input[2]):zero()
   
   -- gradOutput for location constrains 
   self.gradInput = self.gradInput + gradOutput
 
   return {self.gradInput, gradConf} 
end
