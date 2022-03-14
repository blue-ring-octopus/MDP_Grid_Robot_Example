# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 20:49:53 2022

@author: hibad
"""
import numpy as np
import copy
#%% Set-up
gamma=1/2
state_space=[]
for i in range(4):
    for j in range(3):
        if not (i==1 and j==1):
            state_space.append({"state":np.asarray([i,j])})
            
for state in state_space:
    if (state['state'] ==np.array([3,2])).all():
        state['reward']=1
        state['value']=1
    elif (state['state'] ==np.array([3,1])).all():
        state['reward']=-1
        state['value']=-1

    else:
        state['reward']=-0.04
        state['value']=0
        
a=["u","d",'l','r']

for state in state_space:
    state["transition"]=[]
    if (not (state['state'] ==np.array([3,2])).all()) and (not (state['state'] ==np.array([3,1])).all()):
        for other_state in state_space:
            delta=other_state["state"]-state["state"]
            if (delta==[1,0]).all():
                state["transition"].append({"node": other_state, "probability":{"u":0.2/3, "d":0.2/3, "l":0.2/3, "r":0.8}})
            elif(delta==[-1,0]).all():
                state["transition"].append({"node": other_state, "probability":{"u":0.2/3, "d":0.2/3, "l":0.8, "r":0.2/3}})
            elif(delta==[0,1]).all():
                state["transition"].append({"node": other_state, "probability":{"u":0.8, "d":0.2/3, "l":0.2/3, "r":0.2/3}})
            elif(delta==[0,1]).all():
                state["transition"].append({"node": other_state, "probability":{"u":0.2/3, "d":0.8, "l":0.2/3, "r":0.2/3}})

for state in state_space:
    prob={"u":1, "d":1, "l":1, "r":1}
    for neighbor in state["transition"]:
        for action in prob:
            prob[action]-=neighbor["probability"][action]
    if (not (state['state'] ==np.array([3,2])).all()) and (not (state['state'] ==np.array([3,1])).all()):   
        state["transition"].append({"node": state, "probability":prob})       
    
#%% value iteration
for i in range(8):
    for state in state_space:
        max_value=-np.inf
        for action in a:
            v=0
            for neighbor in state["transition"]:
                v+=neighbor["probability"][action]*neighbor['node']["value"]
            if v>=max_value:
                max_value=v
                a_max=action
        state['new_value']=state['reward']+gamma*max_value
        state["policy"]=a_max
        
    for state in state_space:
        state['value']=state['new_value']
    
