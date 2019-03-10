import numpy



class HMM:
    def __init__(self,A,B,Pi):
        self.A=A
        self.B=B
        self.Pi=Pi
    # forward propagation
    def forward(self,O):
        row=self.A.shape[0]
        col=len(O)
        alpha=numpy.zeros((row,col))
        #initial value
        alpha[:,0]=self.Pi*self.B[:,O[0]]
        #update alpha
        for t in range(1,col):
            for i in range(row):
                alpha[i,t]=numpy.dot(alpha[:,t-1],self.A[:,i])*self.B[i,O[t]]
        return alpha

    #back propagation
    def backward(self,O):
        row=self.A.shape[0]
        col=len(O)
        beta=numpy.zeros((row,col))
        #initial value
        beta[:,-1:]=1
        #update beta
        for t in reversed(range(col-1)):
            for i in range(row):
                beta[i,t]=numpy.sum(self.A[i,:]*self.B[:,O[t+1]]*beta[:,t+1])
        return beta

    #Baum-Welch algorithm
    def baum_welch(self,O,e=0.05):

        row=self.A.shape[0]
        col=len(O)

        done=False
        while not done:
            zeta=numpy.zeros((row,row,col-1))
            alpha=self.forward(O)
            beta=self.backward(O)
            #EM(expectation maximization)
            #E step
            for t in range(col-1):
                #denominator part
                denominator=numpy.dot(numpy.dot(alpha[:,t],self.A)*self.B[:,O[t+1]],beta[:,t+1])
                for i in range(row):
                    #numerator part
                    numerator=alpha[i,t]*self.A[i,:]*self.B[:,O[t+1]]*beta[:,t+1]
                    zeta[i,:,t]=numerator/denominator
            gamma=numpy.sum(zeta,axis=1)
            final_numerator=(alpha[:,col-1]*beta[:,col-1]).reshape(-1,1)
            final=final_numerator/numpy.sum(final_numerator)
            gamma=numpy.hstack((gamma,final))
            #M step
            newPi=gamma[:,0]
            newA=numpy.sum(zeta,axis=2)/numpy.sum(gamma[:,:-1],axis=1)
            newB=numpy.copy(self.B)
            b_denominator=numpy.sum(gamma,axis=1)
            temp_matrix=numpy.zeros((1,len(O)))
            for k in range(self.B.shape[1]):
                for t in range(len(O)):
                    if O[t]==k:
                        temp_matrix[0][t]=1
                newB[:,k]=numpy.sum(gamma*temp_matrix,axis=1)/b_denominator
            #ending valve
            if numpy.max(abs(self.Pi-newPi))<e and numpy.max(abs(self.A-newA))<e and numpy.max(abs(self.B-newB))<e:
                done=True 
            self.A=newA
            self.B=newB
            self.Pi=newPi
        return self.Pi


#convert dictionary to matrix
def matrix(X,index1,index2):
    #initialize zero matrix
    m = numpy.zeros((len(index1),len(index2)))
    for row in X:
        for col in X[row]:
            #convert
            m[index1.index(row)][index2.index(col)]=X[row][col]
    return m

if __name__ == "__main__":  
    status=["x1","x2"]
    observations=["1","2","3","4","5","6","7","8"]
    A={"x1":{"x1":0.4,"x2":0.7},"x2":{"x1":0.6,"x2":0.3}}
    B={"x1":{"1":0.1,"2":0.2,"3":0.1,"4":0.1, "5":0.1, "6":0.2, "7":0.1, "8":0.1},"x2":{"1":0.1,"2":0.1,"3":0.2,"4":0.1, "5":0.1, "6":0.1, "7":0.1, "8":0.2}}
    Pi=[0.8,0.2]
    O=[0,1,2,3,4,5,6,7,2,3,4]

    A=matrix(A,status,status)
    B=matrix(B,status,observations)
    hmm=HMM(A,B,Pi)
    print(hmm.baum_welch(O))
