# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 00:13:55 2015

@author: alimehdizadeh
"""

import numpy as np
import math
from scipy import linalg
import scipy
from scipy.integrate import simps

file = open("Output.txt", "w") 
file.close()

def null(A, eps=1e-15):
    """
    For finding kernel of T
    """
    u, s, vh = scipy.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)

# set E_barrier, k, teta_0, E_b, dt, beta

Beta = 1
dt = 0.001
k = 1
teta_0 = 0
dteta = (math.pi/180) * 5

teta = [ 0 , 0 , math.pi, math.pi, 0, math.pi/2, math.pi, -math.pi/2 ]

E_dagger_ab = 0
E_dagger_bc = 0
E_dagger_cd = 0
E_dagger_da = 0

E_dagger_ab = E_dagger_bc = E_dagger_cd = E_dagger_da = 1.0

E_A = 0
E_B = 0
E_C = 0
E_D = 0

E_A = E_C = 0.0
E_B = E_D = 1
   
def States_Energies(Ea, Eb, Ec, Ed, tetaa, tetab, tetac, tetad, k, teta0):
    
    E_a = Ea + 0.5 * k * ( tetaa - teta0 ) * ( tetaa - teta0)
    E_b = Eb + 0.5 * k * ( tetab - teta0 ) * ( tetab - teta0)
    E_c = Ec + 0.5 * k * ( tetac - teta0 ) * ( tetac - teta0)
    E_d = Ed + 0.5 * k * ( tetad - teta0 ) * ( tetad - teta0)
    
    return [ E_a, E_b, E_c, E_d]

Energies = States_Energies(E_A, E_B, E_C, E_D, teta[0], teta[1], teta[2], teta[3], k, teta_0, )


def Barrier_Calculation( E_dagger_ab, E_dagger_bc, E_dagger_cd, E_dagger_da, k, teta0, tetaab, tetabc, tetacd, tetada):
    
    E_b_ab = E_dagger_ab + 0.5*k*(tetaab - teta0) * (tetaab - teta0)
    E_b_bc = E_dagger_bc + 0.5*k*(tetabc - teta0) * (tetabc - teta0) 
    E_b_cd = E_dagger_cd + 0.5*k*(tetacd - teta0) * (tetacd - teta0)   
    E_b_da = E_dagger_da + 0.5*k*(tetada - teta0) * (tetada - teta0) 
        
    
    return [ E_b_ab ,E_b_bc, E_b_cd, E_b_da ]

Barriers = Barrier_Calculation( E_dagger_ab, E_dagger_bc, E_dagger_cd, E_dagger_da, k, teta_0, teta[4], teta[5], teta[6], teta[7])    

    
def Transition_Rates( b_ab, b_bc, b_cd, b_da, Ea, Eb, Ec, Ed, beta ):
    
    T = [  
  	[ 0, 0, 0, 0 ] ,
  	[ 0, 0, 0, 0 ] ,
  	[ 0, 0, 0, 0 ] ,
  	[ 0, 0, 0, 0 ] ,
    ]     
    
    T[0][1] = math.exp(-beta*(b_ab - Ea))
    T[1][0] = math.exp(-beta*(b_ab - Eb))
    T[1][2] = math.exp(-beta*(b_bc - Eb))
    T[2][1] = math.exp(-beta*(b_bc - Ec))
    T[2][3] = math.exp(-beta*(b_cd - Ec))
    T[3][2] = math.exp(-beta*(b_cd - Ed))
    T[3][0] = math.exp(-beta*(b_da - Ed))
    T[0][3] = math.exp(-beta*(b_da - Ea))
    
    maxrate = 0
    for i in range(4):
        for j in range(4):
            if T[i][j] > maxrate :
                maxrate = T[i][j]
                
    for i in range(4):
        for j in range(4):
            dummy = T[i][j]             
            T[i][j] = ( dummy/maxrate)*0.5
        
    return T
    
Transition_Jumps = Transition_Rates( Barriers[0], Barriers[1], Barriers[2], Barriers[3], Energies[0], Energies[1], Energies[2], Energies[3], Beta )

def Force(tetaa, tetab, tetac, tetad, teta0):
   

    Fa = -k*(tetaa - teta0)
    Fb = -k*(tetab - teta0)
    Fc = -k*(tetac - teta0)
    Fd = -k*(tetad - teta0)
    
    return [Fa, Fb, Fc, Fd]
 
F = Force(teta[0],teta[1],teta[2],teta[3], teta_0)
    
def Tran_Matrix_n_step( mylist , n ):
    
    x = np.array(mylist)
    T_power_n = np.array(mylist)
    
    if  n == 0 :
        return [  
  	[ 1, 0, 0, 0 ] ,
  	[ 0, 1, 0, 0 ] ,
  	[ 0, 0, 1, 0 ] ,
  	[ 0, 0, 0, 1 ] ,
    ]     
    
    else :
        for dummy in range(n-1):
            T_power_n = T_power_n * x
            
        return T_power_n 

def teta_assignation(teta0):
    
    if ( (teta0 >= 0) and (teta0 < math.pi/2) ):
        tetaa = tetab = tetaab = 0
        tetabc = math.pi/2
        tetac = tetad = tetacd = math.pi
        tetada = -math.pi/2
        return [ tetaa, tetab, tetac, tetad, tetaab, tetabc, tetacd, tetada]
        
    if ( (teta0 >= math.pi/2) and (teta0 < math.pi) ):
        tetaa = tetab = tetaab = 0
        tetabc = math.pi/2
        tetac = tetad = tetacd = math.pi
        tetada = (3*math.pi)/2
        return [ tetaa, tetab, tetac, tetad, tetaab, tetabc, tetacd, tetada]
        
    if ( (teta0 >= math.pi) and (teta0 < (3*math.pi)/2) ):
        tetaa = tetab = tetaab = 2*math.pi
        tetabc = math.pi/2
        tetac = tetad = tetacd = math.pi
        tetada = (3*math.pi)/2
        return [ tetaa, tetab, tetac, tetad, tetaab, tetabc, tetacd, tetada]
        
    if ( (teta0 >= (3*math.pi)/2)) and (teta0 < 2*math.pi ):
        tetaa = tetab = tetaab = 2*math.pi
        tetabc = (2*math.pi) + math.pi/2
        tetac = tetad = tetacd = math.pi
        tetada = (3*math.pi)/2
        return [ tetaa, tetab, tetac, tetad, tetaab, tetabc, tetacd, tetada]


lambda_dot = []

teta_0 = (math.pi/180) * 0

while ( teta_0 < (math.pi/180) * 360 ):
    
    teta = teta_assignation(teta_0)         
    Energies = States_Energies(E_A, E_B, E_C, E_D, teta[0], teta[1], teta[2], teta[3], k, teta_0)  
    Barriers = Barrier_Calculation( E_dagger_ab, E_dagger_bc, E_dagger_cd, E_dagger_da, k, teta_0, teta[4], teta[5], teta[6], teta[7])    
    Transition_Jumps = Transition_Rates( Barriers[0], Barriers[1], Barriers[2], Barriers[3], Energies[0], Energies[1], Energies[2], Energies[3], Beta )
    
    Transition_Matrix =  [  
  	[ 1 - ( Transition_Jumps[0][1] + Transition_Jumps[0][2] + Transition_Jumps[0][3] )*dt, Transition_Jumps[1][0]*dt, Transition_Jumps[2][0]*dt , Transition_Jumps[3][0]*dt ] ,
  	[ Transition_Jumps[0][1]*dt, 1 - ( Transition_Jumps[1][0] + Transition_Jumps[1][2] + Transition_Jumps[1][3] )*dt, Transition_Jumps[2][1]*dt , Transition_Jumps[3][1] * dt ],
  	[ Transition_Jumps[0][2]*dt , Transition_Jumps[1][2]*dt, 1 - ( Transition_Jumps[2][0] + Transition_Jumps[2][1] + Transition_Jumps[2][3])*dt, Transition_Jumps[3][2]*dt ],
  	[ Transition_Jumps[0][3]*dt, Transition_Jumps[1][3]*dt , Transition_Jumps[2][3]*dt, 1 - ( Transition_Jumps[3][0] + Transition_Jumps[3][1] + Transition_Jumps[3][2])*dt ]
    ]   
    
    dp_Matrix = [  
  	[ Transition_Jumps[0][1] + Transition_Jumps[0][2] + Transition_Jumps[0][3], Transition_Jumps[1][0], Transition_Jumps[2][0], Transition_Jumps[3][0]],
  	[ Transition_Jumps[0][1], Transition_Jumps[1][0] + Transition_Jumps[1][2] + Transition_Jumps[1][3], Transition_Jumps[2][1], Transition_Jumps[3][1]],
  	[ Transition_Jumps[0][2], Transition_Jumps[1][2], Transition_Jumps[2][0] + Transition_Jumps[2][1] + Transition_Jumps[2][3], Transition_Jumps[3][2]],
  	[ Transition_Jumps[0][3], Transition_Jumps[1][3], Transition_Jumps[2][3], Transition_Jumps[3][0] + Transition_Jumps[3][1] + Transition_Jumps[3][2]],
    ]   
  
    y = null( dp_Matrix )
    y[0]=abs(y[0])
    y[1]=abs(y[1])
    y[2]=abs(y[2])
    y[3]=abs(y[3])
    y = y/(y[0]+y[1]+y[2]+y[3])
   
    eq_probs = y
    
    F = Force(teta[0],teta[1],teta[2],teta[3], teta_0)
    avg_force = eq_probs[0]*F[0]+eq_probs[1]*F[1]+eq_probs[2]*F[2]+eq_probs[3]*F[3]
   
    n = 0
    correlation = 1
    Integral = 0
    
    squre_avg = eq_probs[0]*math.pow((F[0]-avg_force),2)+eq_probs[1]*math.pow((F[1]-avg_force),2)+eq_probs[2]*math.pow((F[2]-avg_force),2)+eq_probs[3]*math.pow((F[3]-avg_force),2)
    correlation_0 = math.pow((eq_probs[0]*(F[0]-avg_force)+eq_probs[1]*(F[1]-avg_force)+eq_probs[2]*(F[2]-avg_force)+eq_probs[3]*(F[3]-avg_force)),2)    
    variance = squre_avg - correlation_0
    norm_correlation = 1    
    
    T_n = [  
              	[ 1, 0, 0, 0 ] ,
              	[ 0, 1, 0, 0 ] ,
              	[ 0, 0, 1, 0 ] ,
              	[ 0, 0, 0, 1 ] ,
                ]        
    
    while ( norm_correlation > 0.01 ): 
      
        correlation = 0.0
               
        for i in range(4):
            for j in range(4):
                correlation += eq_probs[i]*T_n[j][i]*(F[i]-avg_force)*(F[j]-avg_force) 
                 
        Integral += correlation
        T_n = np.array(T_n) * np.array(Transition_Matrix)
        n = n + 1
        norm_correlation = correlation / variance 
        
    I = ( Integral / n ) * 1 * (dt * n)      

    print I[0],","    
    file = open("Output.txt", "a")
    file.write(str(I[0]))   
    file.write("\n")
    file.close() 
    
    teta_0 += dteta

