require 'dp'
require 'rnn'
require 'optim'
require '../SpatialGlimpseTwoDimbhwc'
require '../reinforceNormalRangeConfSprd'
require '../recurrentAttentionConfSprd'
require '../HingeEmbeddingCriterionEx'
require '../confusionEx'
require '../imageset2DwithConf'



--require('mobdebug').start() 

version = 1.0

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text('evaluate a VERAM model')
cmd:option('--cuda', true, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:text()
local opt = cmd:parse(arg or {})
if not opt.silent then
  table.print(opt)
end


if opt.cuda then
  require 'cunn'
  require 'optim' 
  cutorch.setDevice(opt.useDevice)
end

modelPath = '../model/ModelNet40-AlexNet-9Views-0.937196InstanceLevelAccuracy-Release.t7'
verammodel = torch.load(modelPath)
verammodel = verammodel:get(1)
print(verammodel) 
verammodel:cuda()

classloss = nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert())
classloss:cuda()

function predictclass()

  local x = batch:inputs():input()
  local y = batch:targets():input()

  if opt.useDevice >= 1 then
    input_data = {x[1]:cuda(),x[2]:cuda()}
  end
  
  verammodel:evaluate()

  -- forward propagate through model
  classoutput = verammodel:forward(input_data) 

  -- measure loss
  classoutputerr = classloss:forward(classoutput[1][1], y)
  
  --print(string.format("class is %d, loss is  %f.... ", tonumber(y[1]), tonumber(classoutputerr)))
  
  classoutputnorm = torch.exp(classoutput[1][1])
  valsingle,indsingle = torch.max(classoutputnorm,2)
  
  isright = 0
  
  if(indsingle[1][1] == y[1]) then
    isright = 1
    end
  conf = classoutputnorm[1][y[1]]
  
  maxidx = indsingle[1][1]

  return isright, conf, maxidx
end

shapelist = {}
file = io.open("../data/ModelNet40_Test_ModelName.txt","r")
for line in file:lines() do
    lineconf = string.gsub(line,'.txt','')
    table.insert(shapelist, lineconf)
end
file:close() 

classnum = 40
classresult = {}
classconf = {}
classmaxidx={}
for i = 1,classnum do
table.insert(classresult, {})
table.insert(classconf, {})
table.insert(classmaxidx, {})
end

--[[data]]--
--when evaluate the model, the train_file and confidence_file is useless, here just for the input integrity of ImageSet2DWithConf
ds = ImageSet2DWithConf({train_file = '../data/ModelNet40_AlexNetFC6_All144Views_Test.h5', validate_file='../data/ModelNet40_AlexNetFC6_All144Views_Test.h5', confidence_file = '../data/ModelNet40_AlexNetFC6_All144Views_TrainData_Confidence.h5', height = 12, width=12, features = 4096, classnum = 40}) 

--local train_set = ds:trainSet()
local valid_set = ds:validSet() 

z = torch.Tensor(classnum,3):zero() 

rightnum = 0
yold = -1
batch = nil
total_val_loss = 0
local sampler = dp.Sampler({batch_size=1}):sampleEpoch(valid_set)
i = 0

local starttime = os.clock();                            
print(string.format("start time : %.4f", starttime));  

while true do
  -- reuse the batch object
  if batch then
    assert(torch.type(batch) == 'dp.Batch')
  end

  batch, nSample, n = sampler(batch)
  if not batch then 
    break 
  end
  
  local y = batch:targets():input()
  if yold ~= y[1] then
    print("];")
  print(string.format("y%d = [", y[1]))
  yold = y[1]
  end 

  
  z[y[1]][1] = z[y[1]][1]+1
  --print(z[1][1])
  i = i+1
  print(shapelist[i])
  
  local isright,modelconf,modelmaxidx = predictclass()
  z[y[1]][2] = z[y[1]][2]+isright
  rightnum = rightnum + isright
  table.insert(classresult[y[1]], isright)
  table.insert(classconf[y[1]], modelconf)
  table.insert(classmaxidx[y[1]], modelmaxidx)
end

local endtime = os.clock();                            
print(string.format("end time   : %.4f", endtime));  
print(string.format("cost time  : %.4f", endtime - starttime));


for i = 1,classnum do
z[i][3] = z[i][2]/z[i][1]
end


print(string.format("acc = %f", rightnum/i))
print(z)
print("+++++classresult++++++++++++++")
print(classresult)
print("+++++classconf++++++++++++++++")
print(classconf)
print("+++++classmaxidx++++++++++++++")
print(classmaxidx)

classlevelacc = 0
for j = 1,classnum do
   z[j][3] = z[j][2]/z[j][1]
   classlevelacc = classlevelacc + z[j][3]/classnum
end

print(z)
print(string.format("total shapes = %d, instance-level accuracy = %f, class-level accuracy = %f,", i, rightnum/i, classlevelacc))


