require 'xlua'
require 'optim'
require 'nn'
dofile './provider.lua'
local c = require 'trepl.colorize'

torch.manualSeed(397)
alpha = .1
opt = lapp[[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 50)          batch size
   -r,--learningRate          (default 1e-2)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default vgg_bn_drop)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default nn)            backend
   --type                     (default cuda)          cuda/float/cl
]]

print(opt)

do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    end
    self.output:set(input)
    return self.output
  end
end

local function cast(t)
   if opt.type == 'cuda' then
      require 'cunn'
      return t:cuda()
   elseif opt.type == 'float' then
      return t:float()
   elseif opt.type == 'cl' then
      require 'clnn'
      return t:cl()
   else
      error('Unknown type '..opt.type)
   end
end

local function noise(input)
    local number = input:size(1)
    local dim = input:size(2)
    local w = input:size(3)
    local h = input:size(4)
    local noise = torch.Tensor(number,dim, w, h):normal(0,0.06)
    return input + noise
end

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
model:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
model:add(cast(dofile('models/'..opt.model..'.lua')))
model:get(2).updateGradInput = function(input) return end

if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.convert(model:get(3), cudnn)
end

print(model)

print(c.blue '==>' ..' loading data')
provider = torch.load 'provider.t7'

--change range to [0,1]
provider.trainData.data = provider.trainData.data:float() 
provider.trainData.labels = provider.trainData.labels:float() 

provider.testData.data = provider.testData.data:float() 
provider.testData.labels = provider.testData.labels:float() 

--confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model:getParameters()


print(c.blue'==>' ..' setting criterion')
criterion = cast(nn.MSECriterion())
criterion2 = cast(nn.SpatialClassNLLCriterion())

print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}


function train()
  model:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  --local targets = cast(torch.FloatTensor(opt.batchSize))
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  error = 0
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    local inputs = provider.trainData.data:index(1,v)
    local inputs_noise = inputs:clone()
    inputs_noise = noise(inputs_noise)
    --targets:copy(provider.trainData.labels:index(1,v))
    local targets = cast(provider.trainData.labels:index(1,v))

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs, outputs_stab = model:forward(inputs, inputs_noise)
      local f = criterion:forward(outputs, targets)
      local f2 = criterion2:forward(outputs, outputs_stab)
	  error = error + f + f2
      local df_do = criterion:backward(outputs, targets)
      local df_do2 = criterion2:backward(outputs, outputs_stab)
      model:backward(inputs, df_do + alpha * df_do2)

      --confusion:batchAdd(outputs, targets)

      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)
  end

  --confusion:updateValids()
  --print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format())
        --confusion.totalValid * 100, torch.toc(tic)))

  --train_acc = confusion.totalValid * 100

  --confusion:zero()
  print("MSE Error: "..error/provider.trainData.data:size(1))
  epoch = epoch + 1
end


function test()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")
  --image.save("test1.png", provider.testData.data[1])
  --image.save("test2.png", provider.testData.data[2])
  local bs = 10
  for i=1,provider.testData.data:size(1),bs do
    local outputs = model:forward(provider.testData.data:narrow(1,i,bs))
   -- print(outputs)
    if outs_score == nil then
        outs_score = outputs:clone()
    else
        outs_score = torch.cat(outs_score, outputs:clone(), 1)
    end
    --confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
	--print(#outputs)
  end

  if outs_score ~= nil then
      local t = {}
      for i = 1, outs_score:size(1) do
         t[i] = {}
         for j = 1, outs_score:size(2) do
            t[i][j] = outs_score[i][j]
         end
      end
      csvigo.save{path = "output_score.csv", data = t}
      outs_score = nil
  end

  

  -- save model every 50 epochs
  if epoch % 100 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(3):clearState())
  end

  --confusion:zero()
end

for i=1,opt.max_epoch do
  train()
  if i % 20 == 0 then
  	test()
  end
end
