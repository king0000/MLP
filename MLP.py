import matplotlib.pyplot as plt
import numpy as np
import math

def hypothesis(X, w_1,b_1,w_2,b_2): 
    h_2 = np.zeros([X.shape[0],10])
    h_1 = np.zeros([X.shape[0],16])
    for i in range(X.shape[0]):
        h_2[i] = softmax(np.dot(relu(np.dot(X[i], w_1)+b_1),w_2)+b_2)
        h_1[i] = relu(np.dot(X[i], w_1)+b_1)  
    return h_1,h_2 
  
# function to compute gradient of error function
def gradient(X, y, w_1,b_1,w_2,b_2): 
    h_1,h_2 = hypothesis(X, w_1,b_1,w_2,b_2)
    grad_2 = np.zeros([X.shape[0],10])
    grad_1 = relu_diff(h_1)
    for i in range(X.shape[0]):
        label = np.argmax(y[i])
        grad_2[i] = -1*h_2[i]
        grad_2[i][label] = (1-h_2[i][label])
    grad_update_2 = np.dot(h_1.transpose(), grad_2) 
    
    tmp = np.dot(w_2,grad_2.transpose())
    grad_1 = grad_1*(tmp.transpose())
    
    grad_update_1 = np.dot(X.transpose(), grad_1)
    
    return grad_update_1,grad_update_2 

  
# function to create a list containing mini-batches 
def create_mini_batches(X, y, batchsize): 
    mini_batches = [] 
    n_minibatches = X.shape[0] // batchsize 
    i = 0
  
    for i in range(n_minibatches): 
        X_mini = X[i*batchsize:(i+1)*batchsize] 
        Y_mini = y[i*batchsize:(i+1)*batchsize] 
        mini_batches.append((X_mini, Y_mini)) 
    if X.shape[0] % batchsize != 0: 
        X_mini = X[i*batchsize: X.shape[0]]
        Y_mini = y[i*batchsize: X.shape[0]]
        mini_batches.append((X_mini, Y_mini)) 
    return mini_batches 
  
def sigmoid(x):
     return 1/(1+np.exp(-x))
    
def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def relu(x):
    return np.maximum(0,x)

def relu_diff(x):
    x[x>0] =1  
    x[x<=0] =0
    return x 
    
def cross_entropy(y_pred,y_label) :
    loss = 0
    datasize = len(y_pred)
    for q in range(10):
        for j in range(datasize):
            if y_label[j][q] == 1:
                loss += -1*math.log(y_pred[j][0][q])
    loss = round(loss/datasize,2)
    return loss

def accuracy(y_pred,y_label) :
    datasize = len(y_pred)
    error = 0
    for j in range(datasize):
        if np.argmax(y_pred[j]) != np.argmax(y_label[j]) :
            error += 1
    acc = round(100*(datasize-error)/datasize ,2)
    return acc

def train(x,y_label,w_1,b_1,w_2,b_2,batch_size,valid_split,epoch,lr):
    datasize = x.shape[0]
    learning_curve_train =[]
    learning_curve_valid =[]
    loss_curve_train =[]
    loss_curve_valid =[]
    trainsize = int(datasize*(1-valid_split))
    validsize = int(datasize*valid_split)
    for i in range(epoch):
        if i != 0:
            per = np.random.permutation(x.shape[0])
            x = x[per, :]
            y_label = y_label[per,:]
        print('epoch : ', i+1)
        mini_batches = create_mini_batches(x, y_label, batch_size) 
        for mini_batch in mini_batches: 
            X_mini, y_mini = mini_batch 
            grad_1,grad_2 = gradient(X_mini, y_mini, w_1,b_1,w_2,b_2)
            w_1 = w_1 + lr *grad_1
            w_2 = w_2 + lr *grad_2
            

        y_predict_train =[]
        y_predict_valid =[]
        for z in range(trainsize):
            y_predict_train.append(softmax(np.dot(relu(np.dot(x[z], w_1)+b_1),w_2)+b_2))
        for z in range(trainsize,datasize):
            y_predict_valid.append(softmax(np.dot(relu(np.dot(x[z], w_1)+b_1),w_2)+b_2))
            
            
        acc_train = accuracy(y_predict_train,y_label[:trainsize])
        acc_valid = accuracy(y_predict_valid,y_label[trainsize:])
        learning_curve_train.append(acc_train)
        learning_curve_valid.append(acc_valid)
        print('accuracy_train : ', acc_train,'%')
        print('accuracy_validation : ', acc_valid,'%')
        loss_train = cross_entropy(y_predict_train,y_label[:trainsize])
        loss_valid = cross_entropy(y_predict_valid,y_label[trainsize:])
        loss_curve_train.append(loss_train)
        loss_curve_valid.append(loss_valid)
        print("loss_train",loss_train)
        print("loss_validation",loss_valid)
        
    e = list(range(1,epoch+1))
    plt.subplot(121)
    plt.plot(e,learning_curve_train,label='train',color ='b')
    plt.plot(e,learning_curve_valid,label ='validation',color ='r')
    plt.xlabel('epoch', fontsize = 12)
    plt.ylabel('accuracy', fontsize = 12)
    plt.legend(loc="lower right")
    plt.subplot(122)
    plt.plot(e,loss_curve_train,label ='train',color ='b')
    plt.plot(e,loss_curve_valid,label='validation',color = 'r')
    plt.xlabel('epoch', fontsize = 12)
    plt.ylabel('loss', fontsize = 12)
    plt.legend(loc="upper right")
    plt.subplots_adjust(wspace =1, hspace =0)
    plt.show()
    
    
    return w_1,b_1,w_2,b_2

