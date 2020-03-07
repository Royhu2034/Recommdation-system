# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import numpy as np

from scipy.optimize import fmin_tnc
import pandas as pd
from scipy import sparse
import numpy as np
import scipy as scp
from sklearn import metrics
#%%
class online_MNL_UCB(object):
    def __init__(self,N,K,X,theta,R,T,p,X_raw):
        # Initialize parameters
        self.N = N # Number of products
        self.K = K # Assortment capacity
        self.X = X # Context data (probably after random projection)
        self.X_raw = X_raw # Raw context data (dimensional)
        self.theta = theta # True parameter
        self.R = R # Bound of parameters
        self.D = len(X[0,:])*self.N # Dimension after random projection
        self.D_raw = len(theta) # Raw data dimension
        self.T = T # Length of time horizon
        self.p = p # Price vector, assumed to be the same for all products
        self.label=np.zeros(self.N)
        self.prob=np.zeros(self.N)
        self.s=list(range(self.K))
    def MNL_revenue(self,v,p):
        # Revenue function
        vp = [a*b for a,b in zip(v,p)]
        return sum(vp)/(1+sum(v))
    
    def Assortment_Opt(self,v,p):
        # Find the optimal assortment given utility vector v and price p
        # Here we assume prices are the same for all products.
        max_S = list(v.argsort()[-self.K:][::-1])
        max_rev = self.MNL_revenue(v[max_S],p[max_S])
        return max_rev,max_S
    

    def cost(self,theta_l,X,Y,S):
        theta = np.zeros([int(self.D/self.N),self.N])
        for i in range(self.N):
            theta[:,i] = theta_l[int(i*self.D/self.N):int((i+1)*self.D/self.N)]
        utility = np.matmul(X,theta)
        exp_utility = np.exp(utility)
        chosen_exp_utility = np.multiply(exp_utility,S)
        chosen_utility = np.multiply(utility,Y)
        
        sum_exp_utility = np.sum(chosen_exp_utility,axis=1)+1
        sum_utility = np.sum(chosen_utility,axis=1)
        
        final_cost = np.sum(np.log(sum_exp_utility))-np.sum(sum_utility)
        print
        return final_cost
    
    def grad(self,theta_l,X,Y,S):
        theta = np.zeros([int(self.D/self.N),self.N])
        for i in range(self.N):
            theta[:,i] = theta_l[int(i*self.D/self.N):int((i+1)*self.D/self.N)]
        utility = np.matmul(X,theta)
        exp_utility = np.exp(utility)
        chosen_exp_utility = np.multiply(exp_utility,S)
        sum_exp_utility = np.sum(chosen_exp_utility,axis=1)+1
        sum_exp_utility = sum_exp_utility.reshape((len(sum_exp_utility),1))
        prob = np.divide(chosen_exp_utility,sum_exp_utility)
        prob_Y = prob-Y
        prob_Y = prob_Y.T
        
#        a = np.array([[1,2],[3,4],[5,6]])
#        a = a.T
        
        grad = np.matmul(prob_Y,X)
        final_grad = np.zeros(self.D)
        for i in range(self.N):
            final_grad[int(i*self.D/self.N):int((i+1)*self.D/self.N)] = grad[i,:]
        return final_grad
    
    def opt(self,theta_t,X,Y,S,epsilon):
        bnds = [(-np.sqrt(1+epsilon)*self.R,np.sqrt(1+epsilon)*self.R)]*self.D    
        theta_t1 = theta_t.copy()
#        print(X,Y,S)
        theta_t1 = fmin_tnc(func=self.cost, x0=theta_t1,
                            fprime=self.grad,args=(X,Y,S),bounds=bnds)
        theta_t1 = theta_t1[0]
        return theta_t1
    def proj_f(self,theta,theta_0,V):
        # Aux function for online learning projection
        diff = theta-theta_0
        f = np.matmul(diff,V)
        f = np.matmul(f,diff.transpose())
        return f
        
    def grad_f(self,theta,theta_0,V):
        # Aux gradient function for online learning projection
        diff = theta-theta_0
        grad = V.dot(diff)
        return grad
        
    
    def Online_Learning(self,theta_t,V_t,S_t,exp_bar_z_t,x_t,Y_t,c,epsilon):
        # This function is to compute the estimator using online learning.
        
        V_t1 = np.zeros([self.D,self.D])
        for i in range(self.N):
            Vti = V_t[i].copy()
            V_t1[int(i*self.D/self.N):int((i+1)*self.D/self.N),int(i*self.D/self.N):int((i+1)*self.D/self.N)] = Vti
        
        grad = np.zeros(self.D)
        for i in S_t: 
            x_tT = np.reshape(x_t,[int(self.D/self.N),1])
            grad[int(i*self.D/self.N):int((i+1)*self.D/self.N)] += (exp_bar_z_t[i]/(1+sum(exp_bar_z_t))-Y_t[i])*x_t
            V_t1[int(i*self.D/self.N):int((i+1)*self.D/self.N),int(i*self.D/self.N):int((i+1)*self.D/self.N)] += np.matmul(x_tT,x_tT.transpose())/len(S_t)
        
        theta_t1 = theta_t.copy()
        for i in S_t:
            V_t1_i = V_t1[int(i*self.D/self.N):int((i+1)*self.D/self.N),int(i*self.D/self.N):int((i+1)*self.D/self.N)]
            grad_i = grad[int(i*self.D/self.N):int((i+1)*self.D/self.N)]
            inv_V_t1_i = np.linalg.inv(V_t1_i)
            theta_t1[int(i*self.D/self.N):int((i+1)*self.D/self.N)] = theta_t[int(i*self.D/self.N):int((i+1)*self.D/self.N)]-c*len(S_t)*inv_V_t1_i.dot(grad_i)
        
        bnds = [(-np.sqrt(1+epsilon)*self.R,np.sqrt(1+epsilon)*self.R)]*self.D                
        theta_t1 = fmin_tnc(func=self.proj_f, x0=theta_t1,
                            fprime=self.grad_f,args=(theta_t1,V_t1),bounds=bnds)
        
        return theta_t1

    
    def original_MNL_bandit(self,alpha,epsilon):
        S_t = list(range(self.K))
        theta_t = np.random.rand(self.D)
        V_t = [(1+epsilon)*np.identity(int(self.D/self.N)) for _ in range(self.N)]        

        track_time = []
        
        cum_reg = 0
        vec_cum_reg = []

        for t in range(self.T):
            start = time.time()
            x_t_raw = self.X_raw[t,:]
            x_t = self.X[t,:] # Context arrive
#            print(t)
            # Decide UCB utility
            v_t = np.ones(self.N)
            true_v_t = np.ones(self.N)
            for i in range(self.N):
                theta_i = theta_t[int(i*self.D/self.N):int((i+1)*self.D/self.N)]
                true_theta_i = self.theta[int(i*self.D_raw/self.N):int((i+1)*self.D_raw/self.N)]
                V_it = V_t[i]
#                print('111',theta_t)
#                print('here',np.exp(np.dot(theta_i,x_t)),alpha*np.sqrt(np.matmul(np.matmul(x_t,np.linalg.inv(V_it)),x_t.transpose())))
                v_t[i] = np.exp(np.dot(theta_i,x_t))+alpha*np.sqrt(np.matmul(np.matmul(x_t,np.linalg.inv(V_it)),x_t.transpose()))                             
                true_v_t[i] = np.exp(np.dot(true_theta_i,x_t_raw))#
            
            # Choose the assortment
            r,S_t = self.Assortment_Opt(v_t,self.p)

            # Calculate the cumulative regret
            opt_rev,opt_S_t = self.Assortment_Opt(true_v_t,self.p)
            
            rev = self.MNL_revenue(true_v_t[S_t],self.p[S_t])
            cum_reg += opt_rev-rev
            
            # Observe customer's choice
            prob = np.zeros(len(S_t)+1)
            for i in range(len(S_t)):
                prob[i] = true_v_t[S_t[i]]/(1+sum(true_v_t[S_t]))
            prob[-1] = 1/(1+sum(true_v_t[S_t]))
            ext_S_t = S_t+[-1]
            cc = np.random.choice(ext_S_t,p=prob)
            Y_t = np.zeros(self.N)
            if cc != -1: Y_t[cc] = 1
            
            
            # Update all parameters
            if t == 0:
                all_S_t = np.zeros(self.N)
                for i in S_t:
                    all_S_t[i] = 1
                all_Y_t = Y_t
            else:
                S_t_add = np.zeros(self.N)
                for i in S_t:
                    S_t_add[i] = 1
                all_S_t = np.vstack([all_S_t, S_t_add])
                all_Y_t = np.vstack([all_Y_t, Y_t])
            
            for i in S_t:
                x_tT = np.reshape(x_t,[int(self.D/self.N),1])
                V_t[i] += np.matmul(x_tT,x_tT.transpose())/len(S_t)
            
            theta_t = self.opt(theta_t,self.X[:(t+1),:],all_Y_t,all_S_t,epsilon)
            
            
            
                
            
            
            vec_cum_reg.append(cum_reg)
            if 1:
                print(t,cum_reg)
            elapsed=(time.time() - start)
            track_time.append(elapsed)
        return track_time
    def MNL_bandit(self,alpha,epsilon):
        # This is the implementation of algorithm 2 with online computation of parameters.
        
        # Initialization
        S_t = list(range(self.K)) # Random choice of assortment
        theta_t = np.random.rand(self.D) # Random choice of parameter estimation
        bar_theta_t = theta_t # Auxillary parameter from online learning
        V_t = [(1+epsilon)*np.identity(int(self.D/self.N)) for _ in range(self.N)] # Store empirical fisher's information matrix V_it
        Xz_t = np.zeros(self.D) # X*z_t
        
        # To store and record regret in each period.
        cum_reg = 0
        cum_opt_rev = 0
        vec_cum_reg = []
        vec_cum_opt_rev = []
        track_time = []
        
        for t in range(self.T):
            start = time.time()
            x_t_raw = self.X_raw[t,:] # Raw context at time t.
            x_t = self.X[t,:] # Context arrive
            
            # Decide UCB utility
            v_t = np.ones(self.N)
            true_v_t = np.ones(self.N)
            
            for i in range(self.N):
                # Estimated hat_theta_i
                theta_i = theta_t[int(i*self.D/self.N):int((i+1)*self.D/self.N)]
                # True theta_i
                true_theta_i = self.theta[int(i*self.D_raw/self.N):int((i+1)*self.D_raw/self.N)]
                V_it = V_t[i]
                # Estimated utility with UCB
                v_t[i] = np.exp(np.dot(theta_i,x_t))+alpha*np.sqrt(np.matmul(np.matmul(x_t,np.linalg.inv(V_it)),x_t.transpose()))                               
                # True utility
                true_v_t[i] = np.exp(np.dot(true_theta_i,x_t_raw))

            # Choose the assortment
            r,S_t = self.Assortment_Opt(v_t,self.p)
            
            # Calculate the cumulative regret
            opt_rev,opt_S_t = self.Assortment_Opt(true_v_t,self.p)
            rev = self.MNL_revenue(true_v_t[S_t],self.p[S_t])
            cum_reg += opt_rev-rev
            cum_opt_rev += opt_rev
            
            # Observe customer's choice
            prob = np.zeros(len(S_t)+1)
            for i in range(len(S_t)):
                prob[i] = true_v_t[S_t[i]]/(1+sum(true_v_t[S_t]))
            prob[-1] = 1/(1+sum(true_v_t[S_t]))
            ext_S_t = S_t+[-1]
            cc = np.random.choice(ext_S_t,p=prob)
            Y_t = np.zeros(self.N)
            if cc != -1: Y_t[cc] = 1
            
            
            # Update all parameters
            bar_z_t = np.zeros(self.D)
            exp_bar_z_t = np.zeros(self.D)
            for i in S_t:
                x_tT = np.reshape(x_t,[int(self.D/self.N),1])
                V_t[i] += np.matmul(x_tT,x_tT.transpose())/len(S_t)
                bar_theta_i = bar_theta_t[int(i*self.D/self.N):int((i+1)*self.D/self.N)]
                bar_z_t[i] = np.dot(bar_theta_i,x_t)
                exp_bar_z_t[i] = np.exp(bar_z_t[i])
            c = 1
            bar_theta_t = self.Online_Learning(bar_theta_t,V_t,S_t,exp_bar_z_t,x_t,Y_t,c,epsilon)
            bar_theta_t = bar_theta_t[0]

            for i in S_t:             
                Xz_t[int(i*self.D/self.N):int((i+1)*self.D/self.N)] += bar_z_t[i]*x_t/len(S_t)
            
            for i in S_t:
                V_t_i = V_t[i]
                inv_V_t_i = np.linalg.inv(V_t_i)
                theta_t[int(i*self.D/self.N):int((i+1)*self.D/self.N)] = inv_V_t_i.dot(Xz_t[int(i*self.D/self.N):int((i+1)*self.D/self.N)])

            vec_cum_reg.append(cum_reg)
            vec_cum_opt_rev.append(cum_opt_rev)
            if 1:
                print(t,cum_reg)
            elapsed=(time.time() - start)
            track_time.append(elapsed)
        return track_time
#%%
class pure_MNL_UCB(object):
    def __init__(self,N,K,X,theta,R,T,p):
        # Initialize parameters
        self.N = N # Number of products
        self.K = K # Assortment capacity
        self.X = X # Context data
        self.theta = theta # True parameter
        self.R = R
        self.D = len(theta)
        self.T = T
        self.p = p
    
    def MNL_revenue(self,v,p):
        # Revenue function
        vp = [a*b for a,b in zip(v,p)]
        return sum(vp)/(1+sum(v))
    
    def Assortment_Opt(self,v,p):
        # Find the optimal assortment given utility vector v and price p
        max_S = list(v.argsort()[-self.K:][::-1])
        max_rev = self.MNL_revenue(v[max_S],p[max_S])
        return max_rev,max_S        
    
    def MNL_bandit(self,alpha,epsilon):
        hat_v = np.ones(self.N) # Store estimated utility in MNL bandit.
        UCB_v = np.ones(self.N) # Store UCB utility in MNL bandit
        T_i = np.zeros(self.N) # Store number of loops of each product.
        
        cum_reg = 0
        vec_cum_reg = []
        
        t = 0 # Track time period
        l = 1 # Track loop
        
        while 1:
            # Choose assortment
            S_l = []
            expl = 0
#            for i in range(self.N): 
#                if T_i[i] < 48*np.log(np.sqrt(self.N)*l+1):
#                    expl = 1
#                    S_l.append(i)
            if len(S_l)>self.K: S_l = S_l[-self.K:]
#                S_l = np.random.choice(S_l,size=self.K,replace=False) 
#                print(S_l)
            if len(S_l) == 0: 
                r,S_l = self.Assortment_Opt(UCB_v,self.p)
                expl = 0
            
            S_l = list(S_l)
            # Offer to customer
            c_t = -2
            num_purchase = np.zeros(self.N)
            while c_t != -1:
                x_t = self.X[t,:] # Context arrive
                true_v_t = np.ones(self.N) # True utility
                for i in range(self.N):
                    true_theta_i = self.theta[int(i*self.D/self.N):int((i+1)*self.D/self.N)]
                    true_v_t[i] = np.exp(np.dot(true_theta_i,x_t))
                # Optimal assortment
                opt_rev,opt_S_t = self.Assortment_Opt(true_v_t,self.p)
                
                # Record regret
                rev = self.MNL_revenue(true_v_t[S_l],self.p[S_l])
                cum_reg += opt_rev-rev
                
                # Observe customer's choice
                prob = np.zeros(len(S_l)+1)
                for i in range(len(S_l)):
                    prob[i] = true_v_t[S_l[i]]/(1+sum(true_v_t[S_l]))
                prob[-1] = 1/(1+sum(true_v_t[S_l]))
                ext_S_t = S_l+[-1]
                
#                print(S_l,ext_S_t,prob)
                c_t = np.random.choice(ext_S_t,p=prob)
                
                if c_t != -1: num_purchase[c_t] += 1
                
                # Time move
                t += 1
                print(t,cum_reg,expl)
                vec_cum_reg.append(cum_reg)
                if t == self.T-1: return vec_cum_reg # Stop whenever time reach end.
            
            # Update data.
            for i in S_l:
                hat_v[i] = (hat_v[i]*T_i[i]+num_purchase[i])/(T_i[i]+1)
                T_i[i] += 1
                UCB_v[i] = hat_v[i]+max(np.sqrt(hat_v[i]),hat_v[i])*np.sqrt(48*np.log(np.sqrt(self.N)*l+1)/T_i[i])+48*np.log(np.sqrt(self.N)*l+1)/T_i[i]
            
            l += 1
                
        
        
        
        return vec_cum_reg

#%%

# Implement pure random policy
class random_MNL_UCB(object):
    def __init__(self,N,K,X,theta,R,T,p):
        # Initialize parameters
        self.N = N # Number of products
        self.K = K # Assortment capacity
        self.X = X # Context data
        self.theta = theta # True parameter
        self.R = R
        self.D = len(theta)
        self.T = T
        self.p = p
    
    def MNL_revenue(self,v,p):
        # Revenue function
        vp = [a*b for a,b in zip(v,p)]
        return sum(vp)/(1+sum(v))
    
    def Assortment_Opt(self,v,p):
        # Find the optimal assortment given utility vector v and price p
        max_S = list(v.argsort()[-self.K:][::-1])     
        max_rev = self.MNL_revenue(v[max_S],p[max_S])
        return max_rev,max_S        
    
    def MNL_bandit(self,alpha,epsilon):
        
        cum_reg = 0
        cum_rev = 0
        vec_cum_reg = []
        
        for t in range(self.T):
            x_t = self.X[t,:] # Context arrive
            true_v_t = np.ones(self.N) # True utility
            for i in range(self.N):
                true_theta_i = self.theta[int(i*self.D/self.N):int((i+1)*self.D/self.N)]
                true_v_t[i] = np.exp(np.dot(true_theta_i,x_t))
            
            # Optimal assortment
            opt_rev,opt_S_t = self.Assortment_Opt(true_v_t,self.p)
            
            # Random selection of assortment
            sz = np.random.choice(range(1,self.K+1))
#            sz = self.K
            S_t = np.random.choice(range(self.N),size=sz,replace=False)
#            S_t = range(self.K)
            # Record regret
            rev = self.MNL_revenue(true_v_t[S_t],self.p[S_t])
            cum_reg += opt_rev-rev
            cum_rev += opt_rev
            vec_cum_reg.append(cum_reg)
            print(t,cum_reg,cum_reg/cum_rev)
        
        return vec_cum_reg
#%%creat x
def generateSparseMatrix(shape, dtype=np.int32, **arguments):


	sparseMat = sparse.lil_matrix(shape, dtype=dtype)

	for row in range(shape[0]):
		whichWay = np.random.choice(range(2), size=1, replace=True)
		
		if whichWay == 0: #选择了X1
			frontIndex = np.random.choice(range(shape[1] // 2), size=arguments['X1frontSparsity'], replace=False)
			behindIndex = np.random.choice(range(shape[1] // 2, shape[1]), size=arguments['X1behindSparsity'], replace=False)
			for num in range(arguments['X1frontSparsity']):
				if (num+1) <= int(0.9 * arguments['X1frontSparsity']):
					sparseMat[row, frontIndex[num]] = 1
				else:
					sparseMat[row, frontIndex[num]] = -1
					
			for num in range(arguments['X1behindSparsity']):
				if (num + 1) <= int(0.1*arguments['X1behindSparsity']):
					sparseMat[row, behindIndex[num]] = 1
				else:
					sparseMat[row, behindIndex[num]] = -1
					
		else: #选择了X2
			frontIndex = np.random.choice(range(shape[1] // 2), size=arguments['X2frontSparsity'], replace=False)
			behindIndex = np.random.choice(range(shape[1] // 2, shape[1]), size=arguments['X2behindSparsity'], replace=False)
			for num in range(arguments['X2frontSparsity']):
				if (num+1) <= int(0.9 * arguments['X2frontSparsity']):
					sparseMat[row, frontIndex[num]] = -1
				else:
					sparseMat[row, frontIndex[num]] = 1
					
			for num in range(arguments['X2behindSparsity']):
				if (num + 1) <= int(0.1*arguments['X2behindSparsity']):
					sparseMat[row, behindIndex[num]] = -1
				else:
					sparseMat[row, behindIndex[num]] = 1
					
	return sparseMat        
#%% creat data
    T = 10000
    N = 10
    K = 4
    d = 6
    D = N*d
    p = np.ones(D)*1
    epsilon=1
    X_raw=(np.random.rand(T,5)*2-1)/np.sqrt(5)
    X_raw = np.concatenate((X_raw, np.ones([T,1])), 1)
    theta = np.random.rand(N*d)
    theta=theta/np.linalg.norm(theta)*10
#%%
theta = pd.read_csv('theta_data.csv')
theta = np.array(theta['0'])
X_raw = pd.read_csv('X_data.csv')
X_raw = np.array(X_raw)
        
   
#%%
# Test Online MNL UCB with parameter tuning
# Dimension 30
import time
start = time.time()
proj_d = 30
# Number of tests
sample_to_run = 5
# List of potential tuning parameters
#tuning = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1]
tuning = [0.1]
for c in tuning:
    reg_table = pd.DataFrame()
    for i in range(sample_to_run):
        
        alpha = c*K*np.sqrt(N)*(np.sqrt(d*np.log(N*T)))
        a = online_MNL_UCB(N,K,X_raw,theta,10,T,p,X_raw)
        reg = a.original_MNL_bandit(alpha,0)
        reg_table[i] = reg       
        
        reg_table.to_csv('Time_MNL_UCB'+'_alpha_'+str(c)+'.csv', index=False)
elapsed=(time.time() - start)
print (elapsed)
        
#%%
# Test pure MNL Bandit
sample_to_run = 3
reg_table = pd.DataFrame()
for i in range(sample_to_run):
    a = pure_MNL_UCB(N,K,X_raw,theta,10,T,p)
    reg = a.MNL_bandit(3,1)
    reg_table[i] = reg

reg_table.to_csv('Regret_Pure_UCB.csv', index=False)    
#%%
#Test random MNL Bandit
sample_to_run = 30
reg_table = pd.DataFrame()
for i in range(sample_to_run):
    a = random_MNL_UCB(N,K,X_raw,theta,10,T,p)
    reg = a.MNL_bandit(3,1)
    reg_table[i] = reg

reg_table.to_csv('Regret_Random_UCB.csv', index=False)   
#%%Test MNL UCB with RP c=0.1
#Dimension 30
import time

sample_to_run = 30

tuning = [0.1,1,0.001,0.0001]
for c in tuning:
    reg_table = pd.DataFrame()
    for i in range(sample_to_run):
        
        alpha = c*K*np.sqrt(N)*(np.sqrt(d*np.log(N*T)))
        a = online_MNL_UCB(N,K,X_raw,theta,10,T,p,X_raw)    
        reg = a.original_MNL_bandit(alpha,0)
        reg_table[i] = reg
        reg_table.to_csv('Regret_MNL_UCB'+'_alpha_'+str(c)+'.csv', index=False)

#for i in range(sample_to_run):
#    
#    alpha = c*K*np.sqrt(N)*(np.sqrt(d*np.log(N*T)))
#    a = online_MNL_UCB(N,K,X_raw,theta,10,T,p,X_raw)    
#    reg = a.original_MNL_bandit(alpha,0)
#    reg_table[i] = reg
#reg_table.to_csv('Regret_MNL_UCB_'+'_alpha_'+str(c)+'.csv', index=False)


#%%

#%%
## Test Online MNL UCB without random projection
## Number of tests
#sample_to_run = 20
#c = 0.1
#if 1 == 1:
#    reg_table = pd.DataFrame()
#    for i in range(sample_to_run):
#        M = np.random.randn(d,proj_d)/np.sqrt(proj_d)
#        X = np.matmul(X_raw,M)
#        alpha = c*K*np.sqrt(N)*(np.sqrt(d*np.log(N*T)))
#            alpha_D = 0.01*K*np.sqrt(N)*(np.sqrt(d*np.log(N*T)))
#        a = online_MNL_UCB(N,K,X_raw,theta,10,T,p,X_raw)
#        reg,opt_rev = a.MNL_bandit(alpha_D,1)
#        reg_table[i] = reg   
#        
#    reg_table.to_csv('Regret_Online_UCB_RP_dim_original'+'_alpha_'+str(c)+'.csv', index=False)
#    rev_table.to_csv('Opt_Revenue_Online_UCB_RP_dim_original'+'_alpha_'+str(c)+'.csv', index=False)    
#%%


#%%
import matplotlib.pyplot as plt
plt.plot(reg)
plt.show()