require 'nn'

local HingeEmbeddingCriterionEx, parent = torch.class('HingeEmbeddingCriterionEx', 'nn.Criterion')

function HingeEmbeddingCriterionEx:__init(margin)
    parent.__init(self)
  
    self.margin = margin or 1
    self.sizeAverage = true
  end 
   
  function HingeEmbeddingCriterionEx:updateOutput(input,y)
     y=-1
     self.buffer = self.buffer or input.new()
     if not torch.isTensor(y) then 
        self.ty = self.ty or input.new():resize(1)
        self.ty[1]=y
        y=self.ty
     end
  
     self.buffer:resizeAs(input):copy(input)
     --self.buffer[torch.eq(y, -1)] = 0
     --self.output = self.buffer:sum()
     
     self.buffer:fill(self.margin):add(-1, input)
     self.buffer:cmax(0)
     --self.buffer[torch.eq(y, 1)] = 0
     --self.output = self.output + self.buffer:sum()
     
     self.output =  self.buffer
     
     if (self.sizeAverage == nil or self.sizeAverage == true) then 
        self.output = self.output:sum()/input:nElement()
     end
  
     
     return self.output
  end
  
  
  function HingeEmbeddingCriterionEx:updateGradInput(input, y)
     y=-1
    -- if not torch.isTensor(y) then self.ty[1]=y; y=self.ty end
     self.gradInput:resizeAs(input):fill(y)
     --self.gradInput[torch.cmul(torch.eq(y, -1), torch.gt(input, self.margin))] = 0
     local tempmargin = input:clone():fill(self.margin)
     indtemp = input.new()
     indtemp = input.new():resize(input:size()):copy(torch.gt(tempmargin, input))
     self.gradInput =  self.gradInput:cmul(indtemp)
     
     if (self.sizeAverage == nil or self.sizeAverage == true) then
        self.gradInput:mul(1 / input:nElement())
     end
        
     return self.gradInput 
  end