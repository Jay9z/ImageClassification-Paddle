#!/usr/bin/env python
# coding: utf-8

import paddle
import paddle.fluid as fluid

from utilis import *
from network import *
from dataset import *


BATCH_SIZE = 32
EPOCH = 80
EPOCH_SIZE = 2160*0.8
BASE_LR = 0.0001  # batch_size=1,GPU=1
RESUME = True

shuffled_reader = fluid.io.shuffle(train_reader,1000)
train_batch_reader = fluid.io.batch(shuffled_reader,BATCH_SIZE)
val_batch_reader = fluid.io.batch(val_reader,1)

## Train model
DEVICE = 'cuda'
gpu_place = fluid.CUDAPlace(0)
logger.info("before training")
with fluid.dygraph.guard(gpu_place):
    #cats = LeNet(num_classes=12)
    #cats = VGG16(num_classes=12)
    cats = ResNet(num_classes=12)
    
    if os.path.exists('result/cats_model.pdparams') and RESUME:
        cats.load_dict(fluid.load_dygraph('result/cats_model')[0])
        logger.info("loading model weight")
    cce = fluid.layers.cross_entropy
    acc = fluid.layers.accuracy

    ## Setup piecewise decay
    step1 = int(EPOCH*EPOCH_SIZE/BATCH_SIZE/1.5)
    step2 = int(EPOCH*EPOCH_SIZE/BATCH_SIZE/1.1)
    boundaries = [step1,step2]
    lr = BASE_LR*BATCH_SIZE
    lr_steps = [lr, lr*0.1, lr*0.01]
    learning_rate = fluid.layers.piecewise_decay(boundaries, lr_steps) 
    ## Setup warmup 
    warmup_steps = min(500,int(step1*0.3))
    start_lr = 0.0
    end_lr = lr
    decayed_lr = fluid.layers.linear_lr_warmup(learning_rate,warmup_steps, start_lr, end_lr)
    ## create an optimizer with customized scheduler
    optimizer=fluid.optimizer.SGDOptimizer(learning_rate=decayed_lr,\
                                        parameter_list=cats.parameters(),\
                                        regularization=fluid.regularizer.L2Decay(1e-4))
    cats.train()
    loss_list = []
    accuracy_max = 0
    accuracy_cur = 0
    for i in range(EPOCH):
        cats.train()
        for j,data in enumerate(train_batch_reader()):
            x = np.array([item[0] for item in data],dtype='float32').reshape((-1,3,224,224))
            y = np.array([item[1] for item in data],dtype='int64').reshape((-1,1))
            x = fluid.dygraph.to_variable(x)
            y = fluid.dygraph.to_variable(y)
            logits = cats(x)
            loss = fluid.layers.softmax_with_cross_entropy(logits=logits,label=y)
            mean_loss = fluid.layers.mean(loss)  
            mean_loss.backward()
            optimizer.minimize(mean_loss)
            cats.clear_gradients()
            logger.info("Epoch {}, step {}: train Loss: {},lr: {:06f}".format(i,j,mean_loss.numpy(),optimizer.current_step_lr())) 

        ## Validation
        cats.eval()
        accuracys = []
        losses = []
        for _,data in enumerate(val_batch_reader()):
            x = np.array([item[0] for item in data],dtype='float32').reshape((-1,3,224,224))
            y = np.array([item[1] for item in data],dtype='int64').reshape((-1,1))
            x = fluid.dygraph.to_variable(x)
            y = fluid.dygraph.to_variable(y)
            logits = cats(x)
            pred = fluid.layers.softmax(logits)
            loss = fluid.layers.softmax_with_cross_entropy(logits,y)
            acc = fluid.layers.accuracy(pred,y)
            losses.append(loss.numpy())
            accuracys.append(acc.numpy())
        accuracy_cur = np.mean(accuracys)
        logger.info("validation Loss: {}, Accuracy: {}".format(np.mean(losses),np.mean(accuracys)))
        
        if accuracy_cur > accuracy_max:
            accuracy_max = accuracy_cur
            fluid.save_dygraph(cats.state_dict(), 'result/cats_model')
            predict_class = []
            for _,data in enumerate(test_reader()):
                x = np.array([item for item in data],dtype='float32').reshape((-1,3,224,224))
                x = fluid.dygraph.to_variable(x)
                logits = cats(x)
                pred = fluid.layers.softmax(logits)
                #y_index = np.argmax(pred.numpy(),axis=1)
                #predict_class.extend(y_index)
                y_index = np.argmax(pred.numpy())
                predict_class.append(y_index)

            # Save predict results
            test_list = np.array(test_list).reshape((-1,1))
            predict_class = np.array(predict_class).reshape((-1,1))
            predict_result = np.concatenate((test_list,predict_class),axis=1)
            if not os.path.exists("result"):
                os.makedirs("result")
            # predict_result.tofile("result/submission.txt",sep=' ')
            df = pd.DataFrame(predict_result,columns=["image_file","id"])
            df.to_csv(f"result/result.csv",sep = ',',header=False,index=False)
    