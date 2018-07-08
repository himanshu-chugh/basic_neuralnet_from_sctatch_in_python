import numpy as np
def activation(k,deriv=False):
    if (deriv==False):
        return (1/(1+np.exp(-k)))
    else:
        return ((k)*(1-k))
    

x=np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]])

y= np.array([
    [1],
    [0],
    [0],
    [1]])


def cost(w1,w2):
    loss=0
    for i in range(len(x)):
            a1=np.array([np.append(x[i],1)])
            
            z2=np.dot(w1,a1.T)
            
            a2=activation(z2)
            
            a2=np.append(a2,1)
            a2=a2.reshape(6,1)
            z3=np.dot(w2,a2)
            a3=activation(z3)
            err3=a3-y[i]
            loss=loss+(err3**2)
    loss=loss*(1/8)
    return loss




np.random.seed(10)
def NN(no_of_itre=10000,lr=.001):
    W1=2*np.random.rand(5,3) - 1
    W2=2*np.random.rand(1,6) - 1
    j=0
    
    while(j!=no_of_itre):
        j+=1
        delta1=np.zeros((5,3),dtype=float)
        delta2=np.zeros((1,6),dtype=float)
        for i in range(len(x)):
            a1=np.array([np.append(x[i],1)])
            
            z2=np.dot(W1,a1.T)
            
            a2=activation(z2)
            
            a2=np.append(a2,1)
            a2=a2.reshape(6,1)
            z3=np.dot(W2,a2)
            a3=activation(z3)
            err3=a3-y[i]
            err2=np.dot(W2.T,err3)*(a2)*(1-a2)
            
            err2=np.delete(err2,5)
            err2=err2.reshape(5,1)
            a1=a1.reshape(1,3)
            
            
            
            delta2=(1/4)*(np.dot(err3,a2.T))+delta2
            delta1=(1/4)*(np.dot(err2,a1))+delta1
        W1=W1-(delta1*lr)
        W2=W2-(delta2*lr)
        #print (j)
        if(j%1000==0):
            print("\n\n\n\n\n\n\n=====================================\nDelta1:\n",delta1,"\n======================================")
            print("Delta2:\n",delta2,"\n---------------------------------------")
            print("cost : \n",cost(W1,W2),"\n------------------------------------------")
            print (j)
    

    return (W1,W2)
    
if(__name__=='__main__'):
    print("Here it begins\n\tDo u want to change some parameters like learning rate or no of iterations?\n\tReply with y/n")
    c=input()
    if (c=='y'):
        noii=input("\nEnter no of iterations :")
        lri=input("\nlearning rate :")
    
        theta1,theta2 = NN(no_of_itre=float(noii),lr=float(lri))
    else:
        theta1,theta2 = NN()    
    
    a1=np.array([np.append([1,1],1)])
            
    z2=np.dot(theta1,a1.T)
            
    a2=activation(z2)
            
    a2=np.append(a2,1)
    a2=a2.reshape(6,1)
    z3=np.dot(theta2,a2)
    a3=activation(z3)
    print(a3)


