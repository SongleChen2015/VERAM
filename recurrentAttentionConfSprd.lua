------------------------------------------------------------------------
--[[ RecurrentAttentionConfSprd ]]-- 
-- module which takes an RNN as argument with other 
-- hyper-parameters such as the maximum number of steps, 
-- action (actions sampling module like ReinforceNormal) and 
------------------------------------------------------------------------
require 'nn'

local RecurrentAttentionConfSprd, parent = torch.class("RecurrentAttentionConfSprd", "nn.AbstractSequencer")

function RecurrentAttentionConfSprd:__init(rnn, action, nStep, hiddenSize)
   parent.__init(self)
   assert(torch.isTypeOf(action, 'nn.Module')) 
   assert(torch.type(nStep) == 'number') 
   assert(torch.type(hiddenSize) == 'table')
   assert(torch.type(hiddenSize[1]) == 'number', "Does not support table hidden layers" )
   
   self.rnn = rnn
   -- we can decorate the module with a Recursor to make it AbstractRecurrent
   self.rnn = (not torch.isTypeOf(rnn, 'nn.AbstractRecurrent')) and nn.Recursor(rnn) or rnn
   
   -- samples an x,y actions for each example
   self.action =  (not torch.isTypeOf(action, 'nn.AbstractRecurrent')) and nn.Recursor(action) or action 
   self.hiddenSize = hiddenSize
   self.nStep = nStep
   
   self.modules = {self.rnn, self.action}
   
   self.output = {}  
   self.actions = {} -- action output
   self.rnnoutput = {} -- rnn output
   
   self.forwardActions = true
   
   self.gradHidden = {}
end

function RecurrentAttentionConfSprd:updateOutput(input)
   self.rnn:forget()
   self.action:forget()
   local nDim = input[1]:dim()
   
   for step=1,self.nStep do
      if step == 1 then
         -- sample an initial starting actions by forwarding zeros through the action
         self._initInput = self._initInput or input[1].new()
         self._initInput:resize(input[1]:size(1),table.unpack(self.hiddenSize)):zero()
         self.actions[1] = self.action:updateOutput({self._initInput,input[2]})
      else
         -- sample actions from previous hidden activation (rnn output)
           self.actions[step] = self.action:updateOutput({self.rnnoutput[step-1], input[2]})
      end
      
      -- rnn handles the recurrence internally
      self.rnnoutput[step] = self.rnn:updateOutput{input[1], self.actions[step]}
      self.output[step] = self.forwardActions and {self.rnnoutput[step], self.actions[step]} or                    self.rnnoutput[step]
   end
   
   return self.output
end

function RecurrentAttentionConfSprd:updateGradInput(input, gradOutput)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
   
   -- back-propagate through time (BPTT)
   for step=self.nStep,1,-1 do
      -- 1. backward through the action layer
      local gradOutput_, gradAction_ = gradOutput[step]
      --for this usage, gradActon_ is valuable
      if self.forwardActions then 
         gradOutput_, gradAction_ = unpack(gradOutput[step])
      else
         -- Note : gradOutput is ignored by REINFORCE modules so we give a zero Tensor instead
         self._gradAction = self._gradAction or self.action.output.new()
         if not self._gradAction:isSameSizeAs(self.action.output) then
            self._gradAction:resizeAs(self.action.output):zero()
         end
         gradAction_ = self._gradAction
      end
      
      if step == self.nStep then
         self.gradHidden[step] = nn.rnn.recursiveCopy(self.gradHidden[step], gradOutput_)
      else
         -- gradHidden = gradOutput + gradAction
         nn.rnn.recursiveAdd(self.gradHidden[step], gradOutput_)
      end
      
      if step == 1 then
         -- backward through initial starting actions
        self.action:updateGradInput({self._initInput, input[2]}, gradAction_)
      else
         local gradAction = self.action:updateGradInput({self.rnnoutput[step-1], input[2]}, gradAction_)
        if(step == self.nStep) then
        end
         self.gradHidden[step-1] = nn.rnn.recursiveCopy(self.gradHidden[step-1], gradAction[1])
      end
      
      -- 2. backward through the rnn layer
      local gradInput = self.rnn:updateGradInput({input[1], self.actions[step]}, self.gradHidden[step])[1]
      if step == self.nStep then
         self.gradInput:resizeAs(gradInput):copy(gradInput)
      else
         self.gradInput:add(gradInput)
      end
   end

   gradConf = input[2].new()
   gradConf:resizeAs(input[2]):zero()
   --self.gradInput = {self.gradInput, gradConf} 
   return {self.gradInput,gradConf}
  
end

function RecurrentAttentionConfSprd:accGradParameters(input, gradOutput, scale)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
   
   -- back-propagate through time (BPTT)
   for step=self.nStep,1,-1 do
      -- 1. backward through the action layer
      local gradAction_ = self.forwardActions and gradOutput[step][2] or self._gradAction
            
      if step == 1 then
         -- backward through initial starting actions
         --self.action:accGradParameters(self._initInput, gradAction_, scale)
         self.action:accGradParameters({self._initInput, input[2]}, gradAction_, scale)
      else
         --self.action:accGradParameters(self.output[step-1], gradAction_, scale)
         self.action:accGradParameters({self.rnnoutput[step-1], input[2]}, gradAction_, scale)
      end
      
      -- 2. backward through the rnn layer
      --self.rnn:accGradParameters({input, self.actions[step]}, self.gradHidden[step], scale)
      self.rnn:accGradParameters({input[1], self.actions[step]}, self.gradHidden[step], scale)
   end
end

function RecurrentAttentionConfSprd:accUpdateGradParameters(input, gradOutput, lr)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    
   -- backward through the action layers
   for step=self.nStep,1,-1 do
      -- 1. backward through the action layer
      local gradAction_ = self.forwardActions and gradOutput[step][2] or self._gradAction
      
      if step == 1 then
         -- backward through initial starting actions
         self.action:accUpdateGradParameters(self._initInput, gradAction_, lr)
      else
         -- Note : gradOutput is ignored by REINFORCE modules so we give action.output as a dummy variable
         self.action:accUpdateGradParameters(self.output[step-1], gradAction_, lr)
      end
      
      -- 2. backward through the rnn layer
      self.rnn:accUpdateGradParameters({input, self.actions[step]}, self.gradHidden[step], lr)
   end
end

function RecurrentAttentionConfSprd:type(type)
   self._input = nil
   self._actions = nil
   self._crop = nil
   self._pad = nil
   self._byte = nil
   return parent.type(self, type)
end

function RecurrentAttentionConfSprd:__tostring__()
   local tab = '  '
   local line = '\n'
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {'
   str = str .. line .. tab .. 'action : ' .. tostring(self.action):gsub(line, line .. tab .. ext)
   str = str .. line .. tab .. 'rnn     : ' .. tostring(self.rnn):gsub(line, line .. tab .. ext)
   str = str .. line .. '}'
   return str
end
