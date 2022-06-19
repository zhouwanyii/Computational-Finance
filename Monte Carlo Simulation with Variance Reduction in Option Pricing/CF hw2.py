#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize


# In[42]:


#question1
def q1(a):
    np.random.seed(10000) 
    z1 = np.random.normal(0, 1, 1000)
    z2 = np.random.normal(0, 1, 1000)
    sigma1=np.sqrt(3)
    sigma2=np.sqrt(5)
    corr=a/(sigma1*sigma2)
    x = sigma1*z1
    y = sigma2*corr*z1+sigma2*np.sqrt(1-corr**2)*z2
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    rho = np.dot(x-x_bar, y-y_bar)/np.sqrt(np.dot(x-x_bar, x-x_bar)*np.dot(y-y_bar, y-y_bar))
    return rho
'%.4f'%q1(-0.7)


# # Question1 
# For n=1000, a=-0.6, with seed = 10000, the corelation rho = -0.1684.

# In[41]:


#question2
def q2(rho):
    np.random.seed(10000) 
    z1 = np.random.normal(0, 1, 10000)
    z2 = np.random.normal(0, 1, 10000)
    sigma1=np.sqrt(1)
    sigma2=np.sqrt(1)
    corr=rho/(sigma1*sigma2)
    x = sigma1*z1
    y = sigma2*corr*z1+sigma2*np.sqrt(1-corr**2)*z2
    z = []
    for i in range(10000):
        z.append(max(0, (x[i]**3+math.sin(y[i])+x[i]**2*y[i])))
    exp = np.mean(z)
    return exp
'%.4f'%q2(0.6)


# # Question 2
# When rho = 0.6, with seed = 10000, E = 1.4922

# In[45]:


#question3
def at(t):
    w_t = np.sqrt(t)*np.random.normal(0, 1, 10000)
    temp = []
    for i in range(10000):
        temp.append(w_t[i]**2+math.sin(w_t[i]))#calculate A(t)
    e = np.mean(temp)#Expectation
    var = np.var(temp)
    return [e,var]

def bt(t):
    w_t = np.sqrt(t)*np.random.normal(0, 1, 10000)
    temp = []
    for i in range(10000):
        temp.append(math.exp(t/2)*math.cos(w_t[i]))#calculate B(t)
    e = np.mean(temp)#Expectation
    var = np.var(temp)
    return [e,var]

#Antithetic Variates Reduction Method
def at_av(t):
    w_t = np.sqrt(t)*np.random.normal(0, 1, 10000)
    w_t_minus = [-i for i in w_t]
    temp = []
    for i in range(10000):
        average = (w_t[i]**2+math.sin(w_t[i])+w_t_minus[i]**2+math.sin(w_t_minus[i]))/2
        temp.append(average)
    e = np.mean(temp)
    var = np.var(temp)
    return [e,var]

def bt_av(t):
    w_t = np.sqrt(t)*np.random.normal(0, 1, 10000)
    w_t_minus = [-i for i in w_t]
    temp = []
    for i in range(10000):
        average = (math.exp(t/2)*math.cos(w_t[i])+math.exp(t/2)*math.cos(w_t_minus[i]))/2
        temp.append(average)
    e = np.mean(temp)
    var = np.var(temp)
    return [e,var]

def q3(seed):
    t=[1,3,5]
    print('Q3 (a) :')
    for i in range(3):
        print('A(',t[i],') is', at(t[i])[0],', B(',t[i],') is', bt(t[i])[0],'.')
    print('Q3 (c) :')
    for i in range(3):
        print('A(',t[i],') is', at_av(t[i])[0],', B(',t[i],') is', bt_av(t[i])[0],'.')
    data = np.array([[t[0],at(t[0])[1],bt(t[0])[1], at_av(t[0])[1],bt_av(t[0])[1]],[t[1],at(t[1])[1],bt(t[1])[1], at_av(t[1])[1],bt_av(t[1])[1]], [t[2],at(t[2])[1],bt(t[2])[1], at_av(t[2])[1],bt_av(t[2])[1]]])
    c = ['t', 'A(t) Before','B(t) Before', 'A(t) After','B(t) After']
    var_comp = pd.DataFrame(data=data,  columns=c, index=t)
    print("\n \nThe variance before and after we adopt Antithetic Variates Reduction Method is ")
    return( var_comp)
q3(10000)


# # Question 3
# (a) With seed = 10000, the resepctive expecations for different t's are: 
# A( 1 ) is 1.0176718764759232 , B( 1 ) is 1.0008450196235845 .
# A( 3 ) is 2.994011044496952 , B( 3 ) is 1.0246665638130696 .
# A( 5 ) is 5.083728238091204 , B( 5 ) is 0.9821612921726012 .
# 

# (b)See attached write up at the end.

# (c) After implementing the Antithetic Variates Reduction Method, with t=5,
# A( 5 ) is 4.944043956830094 , B( 5 ) is 1.1063729102113142 .
# By observing the variance table above about before and after implementing the reduction method, we notice that it does have some effects in reducing the variance, especially when t is small.

# In[48]:


#question4
def q4a(r, sigma, S_0,w_t,t,n): #w_t is the list of random variables, t is the length of time, n is the number of intervals
    S_t = [S_0*math.exp(sigma*i+(r-0.5*sigma**2)*5) for i in w_t]
    ES_t=np.mean(S_t)
    payoff = []
    for i in range(n):
        payoff.append(max((S_t[i]-100), 0))#calculate payoff
    call = math.exp(-r*5)*np.mean(payoff)
    var = math.exp(-r*5)*np.var(payoff)
    return [call,ES_t,var]

def q4b(r, sigma, S_0):
    d1 = (np.log(S_0/100)+(r+0.5*sigma**2)*5)/(sigma*np.sqrt(5))
    d2 = d1 - sigma*np.sqrt(5)
    c_BS = S_0*norm.cdf(d1) - 100*np.exp(-r*5)*norm.cdf(d2)#BS formula
    return c_BS

def q4c(r, sigma, S_0,w_5):#Antithetic Variates reduction technique
    S_5 = [S_0*math.exp(sigma*i+(r-0.5*sigma**2)*5) for i in w_5]
    S_5_minus = [S_0*math.exp(sigma*(-i)+(r-0.5*sigma**2)*5) for i in w_5]
    payoff = []
    for i in range(10000):
        payoff.append((max((S_5[i] - 100), 0) + max((S_5_minus[i] - 100), 0))/2) 
    call = math.exp(-r*5)*np.mean(payoff)
    var = math.exp(-r*5)*np.var(payoff)
    return [call, var]

def q4(r, sigma, S_0):
    np.random.seed(713)
    w_5 = np.sqrt(5)*np.random.normal(0, 1, 10000)
    print("Q4 (a) : The result of the call option price by the Monte Carlo Simulation is", q4a(r, sigma, S_0,w_5,5,10000)[0])
    print("Q4 (b) : The result of the call option price by the Black-Scholes formula is", q4b(r, sigma, S_0))
    print("Q4 (c) : The result of the call option price after using a Antithetic Variates reduction technique is",q4c(r, sigma, S_0,w_5)[0])
    print("The variance before is", q4a(r, sigma, S_0,w_5,5,10000)[2])
    print("The variance after is", q4c(r, sigma, S_0,w_5)[1])
    
q4(0.04, 0.2, 88)


# # Question 4
# (a) By Monte Carlo Simulation, the call option's price is 18.017112538180456.

# (b) By Black-Scholes formula, the call option's price is 18.28376570485581.

# (c) After using Antithetic Variates reduction technique, the call option's price is 18.150630713788775.
# By observing the variance, before is 1207.7511366038145, after is 406.53462669639896, we did see some improvements after implementing the reduction method, variance is reduced. 
# The price is also closer to the formula price rendered by Black-Scholes.

# In[25]:


#question5
def q5(sigma):
    np.random.seed(713)
    S_ta= [88]
    for i in range(10):
        w_t = np.sqrt(i+1)*np.random.normal(0, 1, 1000)
        S_ta.append(q4a(0.04, sigma, 88 ,w_t,i+1,1000)[1])#calculate expectation value as what we did in q4(a)
    t = [i for i in range(11)]
    plt.plot(t, S_ta)
    plt.xlabel('t')
    plt.ylabel('E(St)')
    plt.title("Q5 (a) with σ = "+str(sigma))
    plt.show()
    dt = 10/1000
    t = [i*dt for i in range(1001)] 
    path = pd.DataFrame(columns=['path1', 'path2', 'path3', 'path4', 'path5', 'path6'], index=t)
    for i in range(6):
        w_dt = np.sqrt(dt) * np.random.normal(0, 1, 1000)
        S_tb = [88]
        for j in range(1000):
            S_tb.append(S_tb[j] + S_tb[j] * 0.04 * dt + S_tb[j] * sigma * w_dt[j])
        path.iloc[:, i] = S_tb
        plt.plot(t, path.iloc[:, i], label=path.columns.values.tolist()[i])
    ti = [i for i in range(11)]
    plt.plot(ti, S_ta, label="E(St)")
    plt.title("Q5 (a) and (b) with σ = " + str(sigma))
    plt.legend(loc='upper right')
    plt.show()
q5(0.18)
q5(0.35)


# # Question 5
# As we increase the volatility, both the price and the expected value become more volatile.

# In[26]:


#question6
def q6(n):
    np.random.seed(713)
    dx = 1/n
    x = [i/n for i in range(n+1)]
    fx_dx = []
    for i in range(n):
        fx_dx.append(4*np.sqrt(1-x[i+1]**2)*dx)
    pi_eu = sum(fx_dx)
    print("Q6 (a):  The integral by using the Euler’s discretization scheme is", pi_eu)
    u = np.random.uniform(0, 1, n)
    pi_mc = np.sum([4*np.sqrt(1-x**2)/n for x in u])
    print("Q6 (b):  The integral by using the Monte Carlo Simulation is", pi_mc)
    
    a = [i/100 for i in range(100)]
    var={}
    for i in a:
        var[i]=np.var([np.sqrt(1-j**2)*1/((1-i*j**2)/(1-i/3)) for j in u])
    alpha=list(var.keys())[list(var.values()).index(min(var.values()))]#find a that minimize the variance
    gx = []
    fx=1/(1-0)
    for i in range(n):
        y = np.roots([alpha, 0, -3, (3-alpha)*u[i]])#find the solution of equation to generate the random variable y that have density as t(y)
        y_i = [x for x in y if (x>0 and x<1)][0]
        gx.append(4*np.sqrt(1-y_i**2)/((1-alpha*y_i**2)/(1-alpha/3)))#calculate g(y)f(y)/t(y)
    pi_is = np.mean(gx)#calculate expectation
    print("Q6 (c):  The integral by using the Importance Sampling method is", pi_is)
q6(10000)


# # Question 6
# After implementing the importance sampling method, it becomes close to the true value.
