require 'dp'
require 'rnn'

require 'optim'
require 'SpatialGlimpseTwoDimbhwc'
require 'reinforceNormalRangeConfSprd'
require 'recurrentAttentionConfSprd'
require 'HingeEmbeddingCriterionEx'
require 'confusionEx'
require 'imageset2DwithConf'


--require('mobdebug').start()
 
 version = 1.0 

--[[command line arguments]]-- 
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train VERAM model')
cmd:text('Example:')
cmd:text('$> th ModelNet40AlexTrain.lua > results_ModelNet40AlexTrain.txt') 
cmd:text('Options:')
cmd:option('--xpPath', '', 'path to a previously saved model')
cmd:option('--learningRate', 0.001, 'learning rate at t=0')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 1200, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--decayafter', 600, 'epoch at which linear decayed LR wil start')
cmd:option('--maxEpoch', 1500, 'maximum number of epochs to run')
cmd:option('--maxTries', 450, 'maximum number of epochs to try to find a better local minima')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights') 
cmd:option('--cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--cuda', true, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')

cmd:option('--transfer', 'ReLU', 'activation function')
cmd:option('--uniform', 0.02, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
cmd:option('--weightStd', 0.01, 'initialize parameters using normal distribution') 
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')

--[[ reinforce ]]--
cmd:option('--rewardScale', 1, "scale of positive reward (negative is 0)")
cmd:option('--locatorStd', 0.11, 'stdev of gaussian location sampler (between 0 and 1)')
cmd:option('--stochastic', false, 'Reinforce modules forward inputs stochastically during evaluation') 

--[[ glimpse layer ]]--
cmd:option('--glimpseHiddenSize', 4096, 'size of glimpse hidden layer')
cmd:option('--glimpsePatchHeight', 1, 'size of glimpse patch at highest res (height = width)')
cmd:option('--glimpsePatchWidth', 1, 'size of glimpse patch at highest res (height = width)')
cmd:option('--glimpseScale', 2, 'scale of successive patches w.r.t. original input image')
cmd:option('--glimpseDepth', 1, 'number of concatenated downscaled patches')
cmd:option('--locatorHiddenSize', 128, 'size of locator hidden layer')
cmd:option('--imageHiddenSize', 4224, 'size of hidden layer combining glimpse and locator hiddens')
cmd:option('--glimpseDropout', 0.2, 'ratio of dropout') 

--[[ recurrent layer ]]-- 
cmd:option('--rho', 6, 'back-propagate through time (BPTT) for rho time-steps')
cmd:option('--hiddenSize', 1024, 'number of hidden units used in Simple RNN.')
cmd:option('--FastLSTM', false, 'use LSTM instead of linear layer')

--[[ data ]]--
cmd:option('--trainEpochSize', -1, 'number of train examples seen between each epoch')
cmd:option('--validEpochSize', -1, 'number of valid examples used for early stopping and cross-validation')
cmd:option('--noTest', true, 'dont propagate through the test set')
cmd:option('--overwrite', true, 'overwrite checkpoint')
cmd:option('--save', '', 'overwrite checkpoint')
cmd:option('--seed',1, 'seed for initialization')


cmd:text()
local opt = cmd:parse(arg or {})


--function to run for each parameter config
local fexe = function(opt, globalcurrent, strParam)
   table.print(opt)
       
   --[[observasion subnetwork (rnn input layer)]]
   --encode the view location
   local locationSensor, glimpseSensor, glimpse, rnn, locator, attention, agent
   locationSensor = nn.Sequential()
   locationSensor:add(nn.SelectTable(2))
   locationSensor:add(nn.Linear(2, opt.locatorHiddenSize))  
   locationSensor:add(nn[opt.transfer]())

   --encode the view image 
   opt.glimpsePatchSize = {opt.glimpsePatchHeight, opt.glimpsePatchWidth}
   glimpseSensor = nn.Sequential()
   glimpseSensor:add(SpatialGlimpseTwoDimbhwc(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale):float())
   glimpseSensor:add(nn.Collapse(3))
   glimpseSensor:add(nn.Dropout(opt.glimpseDropout)) 

   --merge the location and view 
   glimpse = nn.Sequential()
   glimpse:add(nn.ConcatTable():add(locationSensor):add(glimpseSensor))
   glimpse:add(nn.JoinTable(1,1))
   glimpse:add(nn.Linear(opt.glimpseHiddenSize+opt.locatorHiddenSize, opt.hiddenSize))
   glimpse:add(nn[opt.transfer]())
   
   --[[RNN subnetwork]]
   glimpse:add(nn.LSTM(opt.hiddenSize, opt.hiddenSize))
   
   
   --[[view estimation subnetwork]]
   locator1 = nn.Sequential()
   locator1:add(nn.SelectTable(1))  --get rnn hidden state
   locator1:add(nn.Linear(opt.hiddenSize, 2))
   locator1:add(nn.ELU()) 
   
   locator2 = nn.Sequential()
   locator2  = nn.SelectTable(2)    --get view confidence
   
   locator = nn.Sequential()
   locator:add(nn.ConcatTable():add(locator1):add(locator2))
   -- sample from normal, uses REINFORCE learning rule
   locator:add(ReinforceNormalRangeConfSprd(opt.locatorStd, opt.stochastic, 0.0834, 1)) 
   
   
   --[[combining observation, view estimation and RNN subnetwork]]
   attention = RecurrentAttentionConfSprd(glimpse, locator, opt.rho, {opt.hiddenSize})

   
   --[[VERAM]]  
   agent = nn.Sequential()
   agent:add(attention)

   
   --[[loss and reward]]
   --the output of rnn is {self.rnnoutput[step], self.actions[step]} 
   --classifier branch, this is the same with before :
   agentclass = nn.Sequential()
   agentclass:add(nn.SelectTable(opt.rho)) -- the rnnoutput last step 
   agentclass:add(nn.SelectTable(1)) -- the rnnoutput last step 
   agentclass:add(nn.Linear(opt.hiddenSize, #ds:classes()))
   agentclass:add(nn.LogSoftMax())

   --add the baseline reward predictor
   seq = nn.Sequential()
   seq:add(nn.Constant(0.5,1))
   seq:add(nn.Add(1))
   concat = nn.ConcatTable():add(nn.Identity()):add(seq)
   concat2 = nn.ConcatTable():add(nn.Identity()):add(concat)

   --output will be : {classpred, {classpred, basereward}}
   agentclass:add(concat2)
   
   --location constrain   
   agentlocation = nn.Sequential()
   concatlocation = nn.ConcatTable()
   for i = 1, opt.rho-1 do
     for j = i+1,opt.rho  do
       locationconstrain = nil
       locationconstrain1 = nn.Sequential()
       locationconstrain1:add(nn.SelectTable(i))  --step
       locationconstrain1:add(nn.SelectTable(2))  --locaton
        
       locationconstrain2 = nn.Sequential()
       locationconstrain2:add(nn.SelectTable(j))  --step
       locationconstrain2:add(nn.SelectTable(2))  --locaton
       
       location = nil
       location = nn.Sequential()
       location:add(nn.ConcatTable():add(locationconstrain1):add(locationconstrain2))
       location:add(nn.PairwiseDistance(2))  
       
       concatlocation:add(location)
     end
   end 
   agentlocation:add(concatlocation)
   
   --the output is {{classpred, {classpred, basereward}}, {locpar1,locpara2,locpar3......}}
   agent:add(nn.ConcatTable():add(agentclass):add(agentlocation))  

   --[[parameter initialization]]
   torch.manualSeed(opt.seed) 
   if opt.uniform > 0 then
      for k,param in ipairs(agent:parameters()) do   
        --param:uniform(-opt.uniform, opt.uniform)
        param:normal(0, opt.weightStd)         
      end
   end


   parametersCheck, gradParametersCheck = agent:getParameters()
   print(string.format('parameter num: %d', (parametersCheck:size())[1]))
   
   --[[Propagators]]--
   opt.decayFactor = (opt.minLR - opt.learningRate)/(opt.saturateEpoch - opt.decayafter)
   
   lossclass = nn.ParallelCriterion(true)
    :add(nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert()), 1.0) 
    :add(nn.ModuleCriterion(nn.VRClassReward(agent, opt.rewardScale), nil, nn.Convert())) 
         
   losslocation = nn.ParallelCriterion(true)
   for i = 1, opt.rho-1 do
     for j = i+1,opt.rho  do 
       --loss of view loction constrains, 1/12 = 0.0833
         losslocation:add(nn.ModuleCriterion(HingeEmbeddingCriterionEx(0.0833), nil, nn.Constant(-1)))
     end
   end
   
   losscat =  nn.ParallelCriterion(true):add(lossclass):add(losslocation)
   
   --save the best model
   local maxinstanceacc = -1
   local maxclassacc = -1
   local maxinstancemodel = nil
   local maxclassmodel = nil
   local starttime = os.date("%Y%m%d%H%M%S")

   train = dp.Optimizer{
      loss =losscat,
      epoch_callback = function(model, report) --call by each epoch 
         if report.epoch > opt.decayafter then
           opt.learningRate = opt.learningRate + opt.decayFactor
           opt.learningRate = math.max(opt.minLR, opt.learningRate)
         end
         if not opt.silent then
           print("learningRate", opt.learningRate)
         end
         if report.epoch > 0 then
           trainaccInstance = report.optimizer.feedback.confusion.accuracy*100
           trainaccClass = report.optimizer.feedback.confusion.avg_per_class_accuracy*100
           valaccInstance = report.validator.feedback.confusion.accuracy*100
           valaccClass = report.validator.feedback.confusion.avg_per_class_accuracy*100
           
           if (valaccClass > maxclassacc) then
              maxclassacc = valaccClass
              maxclassmodel = model:clone()
           end
           
           if (valaccInstance > maxinstanceacc) then
              maxinstanceacc = valaccInstance
              maxinstancemodel = model:clone()
           end

         end
      end,
      callback = function(model, report)  --call by each batch
        if opt.cutoffNorm > 0 then
          local norm = model:gradParamClip(opt.cutoffNorm) 
            opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
            if opt.lastEpoch < report.epoch and not opt.silent then
              print("mean gradParam norm", opt.meanNorm)
            end
        end
        model:updateGradParameters(opt.momentum) 
        model:updateParameters(opt.learningRate)
        model:maxParamNorm(opt.maxOutNorm)
        model:zeroGradParameters()
      end,
      feedback = ConfusionEx{output_module=nn.Sequential():add(nn.SelectTable(1)):add(nn.SelectTable(1))},
      sampler = dp.ShuffleSampler{
         epoch_size = opt.trainEpochSize, batch_size = opt.batchSize
      },
      progress = opt.progress
   }
   
   
   valid = dp.Evaluator{
      feedback = ConfusionEx{output_module=nn.Sequential():add(nn.SelectTable(1)):add(nn.SelectTable(1))},
      sampler = dp.Sampler{epoch_size = opt.validEpochSize, batch_size = opt.batchSize},
      progress = opt.progress
   }
   if not opt.noTest then
      tester = dp.Evaluator{
         feedback = ConfusionEx{output_module=nn.SelectTable(1)},
         sampler = dp.Sampler{batch_size = opt.batchSize}
      }
   end
   
   --[[Experiment]]
   xp = dp.Experiment{
      model = agent,
      optimizer = train,
      validator = valid,
      tester = tester,
      observer = {
        ad,
        dp.FileLogger(),
        dp.EarlyStopper{
          max_epochs = opt.maxTries,
          --define the accuracy field in the report
          error_report={'validator','feedback','confusion','avg_per_class_accuracy'},
          maximize = true
         }
      },
      random_seed = os.time(),
      max_epoch = opt.maxEpoch
   }
   
   --[[GPU or CPU]]
   if opt.cuda then
      print"Using CUDA"
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(opt.useDevice)
      xp:cuda()
   else
      xp:float()
   end
   
   xp:verbose(not opt.silent)
   if not opt.silent then
      print"Agent :"
      print(agent)
   end
   
   xp.opt = opt
   
   if checksum then
      assert(math.abs(xp:model():parameters()[1]:sum() - checksum) < 0.0001, "Loaded model parameters were  changed???")
   end
   xp:run(ds)
   
   --save the best instance-level and class-level model
   maxinstanceacc = maxinstanceacc/100 
   maxclassacc = maxclassacc/100
   local maxinstancepath = string.format('./maxinstance-%f-%s-%s.t7', maxinstanceacc, strParam, starttime)
   local maxclassspath = string.format('./maxclass-%f-%s-%s.t7', maxclassacc, strParam, starttime)
   
   print('train finish,maxinstanceacc-'..maxinstanceacc..', maxclassacc-'..maxclassacc)
   print('++++++save max instance accuracy model to  '..maxinstancepath)
   torch.save(maxinstancepath, maxinstancemodel)
   print('******save max class accuracy model to  '..maxclassspath)
   torch.save(maxclassspath, maxclassmodel)
   
   
   return maxclassacc, maxclassmodel

end

--[[dataset]]
ds = ImageSet2DWithConf({train_file = './data/ModelNet40_AlexNetFC6_All144Views_Train.h5', validate_file='./data/ModelNet40_AlexNetFC6_All144Views_Test.h5', confidence_file = './data/ModelNet40_AlexNetFC6_All144Views_TrainData_Confidence.h5', height = 12, width=12, features = 4096, classnum = 40})

--min learning rate
_learningRate = {0.001}
_minLR = {0.00001}  
-- init parameters
_weightStd = {0.01}  
--dropout
_glimpseDropout = {0.2}
--hiddenSize
_hiddenSize = {1024}
--locatorStd
_locatorStd = {0.11}
--rho
--_rho={2,3,4,5,6,7,8,9,10,11,12}
_rho={9}  


totalSuperParam = #_learningRate*#_minLR*#_weightStd*#_glimpseDropout*#_hiddenSize*#_locatorStd*#_rho

 currentProgress = 0
 globalBestAcc = 0
 globalBestModel = nil
 globalBestParam = nil
 for h,learningRate in pairs(_learningRate) do
 for i,minLR in ipairs(_minLR) do 
 for j,weightStd in ipairs(_weightStd) do 
 for l,glimpseDropout in ipairs(_glimpseDropout) do 
 for m,hiddenSize in ipairs(_hiddenSize) do 
 for n,locatorStd in ipairs(_locatorStd) do
 for n,rho in ipairs(_rho) do
   currentProgress = currentProgress + 1
   print('+++++current global progress:'..currentProgress..'/'..totalSuperParam..'++++++')
   opt.learningRate = learningRate
   opt.minLR = minLR
   opt.weightStd = weightStd
   opt.glimpseDropout = glimpseDropout
   opt.hiddenSize = hiddenSize
   opt.locatorStd = locatorStd
   opt.rho = rho
  
   strParam = string.format('m40AlexFC6LSTM-lr%f-minlr%f-wstd%.2f-drop%.2f-hid%d-lstd%.2f-rho%d',opt.learningRate, opt.minLR, opt.weightStd,opt.glimpseDropout,opt.hiddenSize, opt.locatorStd,opt.rho)
   
   bestAcc = fexe(opt, currentProgress, strParam)
   if( bestAcc > globalBestAcc) then
      globalBestAcc = bestAcc
      globalBestParam = strParam
   end
   
   print('param-'..strParam..', bestAcc:'..bestAcc..', global bestAcc:'..globalBestAcc)
   if globalBestParam then
       print('global best param-'..globalBestParam..', global bestAcc:'..globalBestAcc)
   end
   
   collectgarbage()
 end
 end
 end
 end
 end 
 end
 end 
 
 
 print( 'all done, the best performance is:'..globalBestAcc)
 

