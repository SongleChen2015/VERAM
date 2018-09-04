require 'loadcaffe'
require 'cutorch'
require 'cunn'
require 'cudnn'
 
CaffeLoader = {}

function CaffeLoader.load(prototxt, caffemodel_path, torch_model_path)

    local returnmodel = nil
	
	local caffemodel = loadcaffe.load(prototxt,caffemodel_path,'cudnn')
    
    returnmodel = caffemodel
	
	-- Creating the sequential model from table of moduels
	-- Copying caffe weight models
    if torch_model_path  then
        local opt = {}
  	    opt.nclass = 20
        local torch_model = dofile(torch_model_path)(opt)
	    local torch_parameters = torch_model:getParameters()
	    local caffeparameters = caffemodel:getParameters()
        torch_parameters:copy(caffeparameters)
        returnmodel = torch_model
     end
	return returnmodel
end

return CaffeLoader
