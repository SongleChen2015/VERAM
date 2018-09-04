------------------------------------------------------------------------
--[[ ImageSet2DWithConf ]]--
------------------------------------------------------------------------
require 'hdf5'
require 'dp'

local ImageSet2DWithConf, DataSource = torch.class("ImageSet2DWithConf", "dp.DataSource")
ImageSet2DWithConf.isImageSet2DWithConf = true

ImageSet2DWithConf._name = 'ImageSet2DWithConf'
ImageSet2DWithConf._image_size = {12, 4096, 1}
ImageSet2DWithConf._image_axes = 'bhwc'
ImageSet2DWithConf._feature_size = 1*4096*12


ImageSet2DWithConf._classes = {} 
function ImageSet2DWithConf:__init(config) 
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, load_all, input_preprocess, target_preprocess
   args, self._valid_ratio, self._train_file, self._validate_file,self._conf_file, self._test_file, 
         self._data_path, self._scale, self._binarize, self._shuffle,
         self._download_url, load_all, input_preprocess, 
         target_preprocess, self._height, self._width, self._features, self._classnum
      = xlua.unpack(
      {config},
      'ImageSet2DWithConf', 
      'Handwritten digit classification problem.' ..
      'Note: Train and valid sets are already shuffled.',
      {arg='valid_ratio', type='number', default=1/6,
       help='proportion of training set to use for cross-validation.'},
      {arg='train_file', type='string', default='train.th7',
       help='name of training file'},
      {arg='validate_file', type='string', default='validate.th7',
       help='name of validate file'},
      {arg='confidence_file', type='string', default='',
       help='confidence file'},
      {arg='test_file', type='string', default='test.th7',
       help='name of test file'},
      {arg='data_path', type='string', default=dp.DATA_DIR,
       help='path to data repository'},
      {arg='scale', type='table', 
       help='bounds to scale the values between. [Default={0,1}]'},
      {arg='binarize', type='boolean', 
       help='binarize the inputs (0s and 1s)', default=false},
      {arg='shuffle', type='boolean', 
       help='shuffle different sets', default=false},
      {arg='download_url', type='string',
       default='',
       help='URL from which to download dataset if not found on disk.'},
      {arg='load_all', type='boolean', 
       help='Load all datasets : train, valid, test.', default=true},
      {arg='input_preprocess', type='table | dp.Preprocess',
       help='to be performed on set inputs, measuring statistics ' ..
       '(fitting) on the train_set only, and reusing these to ' ..
       'preprocess the valid_set and test_set.'},
      {arg='target_preprocess', type='table | dp.Preprocess',
       help='to be performed on set targets, measuring statistics ' ..
       '(fitting) on the train_set only, and reusing these to ' ..
       'preprocess the valid_set and test_set.'} ,
       {arg='height', type='number', default=0,
       help='nubmer of height.'},
      {arg='width', type='number', default=0,
       help='nubmer of width.'},
      {arg='features', type='number', default=0,
       help='feature length of a view.'}, 
      {arg='classnum', type='number', default=10,
       help='number of classes'}   
     
   )
   if (self._scale == nil) then
      self._scale = {0,1} 
   end
   
   ImageSet2DWithConf._image_size = {self._height, self._width, self._features}
   ImageSet2DWithConf._feature_size = self._height*self._width*self._features

   for i= 1,self._classnum do
     ImageSet2DWithConf._classes[i] = i
   end
   
   
   if load_all then
      self:loadTrainValid()
      --self:loadTest()
   end
   DataSource.__init(self, {
      train_set=self:trainSet(), valid_set=self:validSet(),
      test_set=nil, input_preprocess=input_preprocess,
      target_preprocess=target_preprocess
   })
end

function ImageSet2DWithConf:loadTrainValid()
   --Data will contain a tensor where each row is an example, and where
   --the last column contains the target class.
   
   -- train
   --local data = self:loadhdf5data('/media/iot/mydisk1/benchmark/3dshape/modelnet40v1_hdf5_train_reverse.h5', 40)
   print('load confidence data...')
   conftraindata = self:loadhdf5dataConf()
   print('load train file '..self._train_file)
   local datatrain = self:loadhdf5data(self._train_file, self._classnum)
   datatrainmerge = {datatrain[1], conftraindata}
   self:trainSet(
       self:createDataSet(datatrainmerge, datatrain[2], 'train')
   )
   
   
   --valid
   print('load validate file '..self._validate_file)
   local dataval = self:loadhdf5data(self._validate_file, self._classnum)
   start = 1
   sizeval = (#(dataval[1]))[1]
   --just for get the same input foramt
   confvaldata = conftraindata:clone():narrow(1, 1, sizeval):zero()
   datavalmerge = {dataval[1], confvaldata}
   self:validSet(
      self:createDataSet(datavalmerge, dataval[2], 'valid')
   )
  
   return self:trainSet(), self:validSet()
end

function ImageSet2DWithConf:loadTest()
   local test_data = self:loadData(self._test_file, self._download_url)
   self:testSet(
      self:createDataSet(test_data[1], test_data[2], 'test')
   )
   return self:testSet()
end

--Creates an Dataset out of inputs, targets and which_set
function ImageSet2DWithConf:createDataSet(inputs, targets, which_set)
   if self._shuffle then
      local indices = torch.randperm(inputs:size(1)):long()
      inputs = inputs:index(1, indices)
      targets = targets:index(1, indices)
   end
   if self._binarize then
      DataSource.binarize(inputs, 128)
   end

   -- construct inputs and targets dp.Views 
   local input_v, target_v = dp.ListView({dp.ImageView(), dp.ImageView()}), dp.ClassView()
   input_v:forward({self._image_axes, self._image_axes}, inputs)
   target_v:forward('b', targets)
   target_v:setClasses(self._classes)
   -- construct dataset
   local ds = dp.DataSet{inputs=input_v,targets=target_v,which_set=which_set}
   ds:ioShapes('bhwc', 'b')
   return ds
end

function ImageSet2DWithConf:loadData(file_name, download_url)
   local path = DataSource.getDataPath{
      name=self._name, url=download_url, 
      decompress_file=file_name, 
      data_dir=self._data_path
   }
   -- backwards compatible with old binary format
   local status, data = pcall(function() return torch.load(path, "ascii") end)
   if not status then
      return torch.load(path, "binary")
   end
   return data
end


function ImageSet2DWithConf:loadhdf5data(filepath, classnumber)

    local imgdata = torch.Tensor()
    local labledata = torch.Tensor()
    local hdf5_data = hdf5.open(filepath, 'r')
    local tclasssize = {}
    local classsamplenum = torch.Tensor(classnumber)
    for i = 1, classnumber do
      local dset_name = string.format('data/%d', i)
      tclasssize[i] = hdf5_data:read(dset_name):dataspaceSize()
      classsamplenum[i] =  tclasssize[i][1]
    end
    local totalsize  = tclasssize[1]
     totalsize[1] = classsamplenum:sum()
     imgdata = torch.FloatTensor(torch.LongStorage(totalsize))
     labledata = torch.Tensor(totalsize[1])
     local startpos = 1
     for i = 1, classnumber do
        print('load calss '..i..'...')
        local dset_name = string.format('data/%d', i)
        local samplenum = hdf5_data:read(dset_name):dataspaceSize()[1]
        local classdata = hdf5_data:read(dset_name):partial({1, samplenum}, {1, self._height}, {1, self._width},{1,self._features})
        if torch.type(classdata) ~= 'torch.FloatTensor' then
            classdata = classdata:float()
        end
        
        imgdata:narrow(1, startpos, samplenum):copy(classdata)
        labledata:narrow(1, startpos, samplenum):fill(1*i)
        startpos = startpos + samplenum
      end
      
    local viewData =  imgdata:view(torch.LongStorage{(#imgdata)[1], (#imgdata)[2], (#imgdata)[3], (#imgdata)[4]})
    local viewLable = labledata:view(torch.LongStorage{(#labledata)[1]})
    
    return {viewData, viewLable}
end


 function ImageSet2DWithConf:loadhdf5dataConf()
    classnumber = self._classnum 
    filepath = self._conf_file
    local conflength = 1
    local confdata = torch.Tensor()
    local labledata = torch.Tensor()
    local hdf5_data = hdf5.open(filepath, 'r')
    for i = 1, classnumber do
        local dset_name = string.format('data/%d', i)
        local samplenum = hdf5_data:read(dset_name):dataspaceSize()[1]
        local confinput = hdf5_data:read(dset_name):partial({1, samplenum}, {1, self._height}, {1, self._width},{1, conflength})       
        if i == 1 then
           confdata = confinput:clone()
        else
           confdata = torch.concat({confdata,confinput},1)
       end        
    end
    
    
    local viewData =  confdata:view(torch.LongStorage{(#confdata)[1], (#confdata)[2], (#confdata)[3], (#confdata)[4]})
    
    return viewData
end


