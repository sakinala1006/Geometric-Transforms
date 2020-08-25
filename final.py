#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 22:10:56 2020

@author: saisrinijasakinala
"""

import numpy as np
import imageio
from shapely.geometry import Point, Polygon
import tqdm



def calculateInitP(sourceCord,targetCord):
    
    #number of corners
    dim = len(targetCord)
    
    Xt  = np.zeros((3,dim),dtype='float')
    
    Xs  = np.zeros((3,dim),dtype='float')
    
    #target and source co-ordinates are put into Xt and Xs such that Xt=[[x] and Xs=[[x']
    #                                                                    [y]         [y']
    #                                                                    [1]]        [1]]
    for i in range(dim):
        Xt[0][i] = targetCord[i][0]
        Xt[1][i] = targetCord[i][1]
        Xs[0][i] = sourceCord[i][0]
        Xs[1][i] = sourceCord[i][1]
        Xt[2][i] = Xs[2][i] = 1
        
    #calculating deltaZ
    deltaZ=np.matmul(np.array([[1,0,0],[0,1,0]]),np.subtract(Xs,Xt))
    
    #P1 and P2 matrices for homography
    P1 = np.array([[1,0,0,0,0,0,-1,0],[0,1,0,0,0,0,0,-1],[0,0,1,0,0,0,0,0]])
    P2 = np.array([[0,0,0,1,0,0,-1,0],[0,0,0,0,1,0,0,-1],[0,0,0,0,0,1,0,0]])

    #J1 = XT.P1
    J1 = np.matmul(Xt.T,P1)
    
    #Using the additive algorithm for direct intensity-based registration, D is assumed to be 1
    # xi^ and yi^ are taken as x and y directly
    for i in range(J1.shape[0]):
        J1[i][7]=-(J1[i][6]*J1[i][7])
        J1[i][6]=-(J1[i][6]**2)
    
    #J2 = XT.P2
    J2 = np.matmul(Xt.T,P2)
    
    for i in range(J2.shape[0]):
        J2[i][7]=-(J2[i][6]*J2[i][7])
        J2[i][6]=-(J2[i][6]**2)
    
    #Calculating r matrix
    deltaX = np.append(np.vstack(deltaZ[0]), np.vstack(deltaZ[1]), axis=0)
    
    #Calculating Jacobian matrix
    J=np.vstack([J1,J2])
    
    #Calculating Hessian matrix
    A=np.matmul(J.T,J)
    
    #Calculating b matrix
    b=np.matmul(J.T,deltaX)

    #Initial Parameter estimate
    P = np.matmul(np.linalg.inv(A),b)
    
    return P



def getConfig(fs,ft):
    
    #get lines from config files
    fs1=fs.readlines()
    ft1=ft.readlines()
    
    sourceCord=[]
    sourcePath=fs1[0].replace('\n','')
    source=imageio.imread(sourcePath) #getting the relative path to the source image
    
    for i in range(1,len(fs1)):
        temp=fs1[i].replace('\n','')
        x,y = temp.split(' ',1)
        x,y = int(x),int(y)
        sourceCord.append((x,y)) #getting the corner co-ordinates of source image
        
    targetCord=[]
    targetPath=ft1[0].replace('\n','')
    target=imageio.imread(targetPath) #getting the relative path to the target image
        
    damp=float(ft1[2].replace('\n','')) #getting the dampening factor of target image

    for i in range(3,len(ft1)):
        temp=ft1[i].replace('\n','')
        x,y = temp.split(' ',1)
        x,y = int(x),int(y)
        targetCord.append((x,y)) #getting the corner co-ordinates of target image
    
    needHinit=ft1[1].replace('\n','') 
    
    #checking whether initial parameter value needs to be estimated or not
    if needHinit == 'no':
        P = np.array([[1,0,0,0,1,0,0,0]],dtype=float).reshape(8,1) #if no, return identity matrix as P
        return source,target,sourceCord,targetCord,P,damp
        
    if needHinit=='yes':
        P=calculateInitP(sourceCord,targetCord) #if yes, call calculateInitP to estimate initial parameters
        return source,target,sourceCord,targetCord,P,damp
    
    
    
def estimateParameters(sourceCord,targetCord,P,damp):
    
    res=100000000000
    
    #previous residual
    resprev=res+1
    while res<resprev and res>1: #enter if the current residual is less than the previous one
        resprev=res
        res=0
        print("Entering the loop")
        
        A , b , p = np.zeros((8,8)) , np.zeros((8,1)) , np.zeros((8,1)) #initializing A,b and temporary p matrices
        
        for X,Xi in zip(sourceCord,targetCord):
            
            x,y=X #source co-ordinates
            xi,yi = Xi #target co-ordinates
            
            #initial H matrix(3x3) in which h22 is taken as 1.
            H = np.append(P,1)
            H = np.reshape(H,(3,3))
            
            #target homogenous coordinates matrix
            targetx=np.array([[xi],
                        [yi],
                        [1]])
    
            #calculating [[a]
            #             [b]
            #             [D]]
            temp=np.matmul(H,targetx)

            D=temp[2][0]
            
            #calculating Jacobian matrix
            J=np.array([[xi/D , yi/D , 1/D , 0 , 0 , 0 , -xi*temp[0][0]/D , -yi*temp[1][0]/D],
                        [0 , 0 , 0 , xi/D , yi/D , 1/D , -xi*temp[1][0]/D , -yi*temp[0][0]/D]])
    
            #calculating displacement matrix
            deltaX = np.array([[x - temp[0][0]/D],
                             [y - temp[1][0]/D]])
            
            #calculating Hessian -> A = JT.J
            A+= np.matmul(J.T,J)
            
            #calculating b matrix -> b = JT.deltaX
            b+= np.matmul(J.T,deltaX)
            
            #calculating residual -> residual = deltaXT - deltaX
            res+= np.matmul(deltaX.T , deltaX)

        print(res)

        #Using dampening factor, calculate P -> P = (A^-1).b
        p = np.matmul(np.linalg.inv(np.add(A,damp*(np.diag(np.diag(A))))),b)
        P+= p

    #reshaping P and returning H
    H=np.append(P,1)
    H=np.reshape(H,(3,3))
    return H



def main():
    
    #file pointers for source and target configuration files
    fs=open(input("Enter source config file:"))
    ft=open(input("Enter target config file:"))
    
    #get source and target images, source and target co-ordinates, initial P and dampening value
    source, target, sourceCord, targetCord, P, damp = getConfig(fs,ft)
    
    #get the final Parameter matrix
    H = estimateParameters(sourceCord,targetCord,P,damp)
    
    #creating a polygon using target co-ordinates
    shapee=Polygon(targetCord)
    
    #iterating through every pixel in the target image
    for i in tqdm.tqdm(range(target.shape[0])):
        for j in range(target.shape[1]):
            pointt = Point(j,i)
            if pointt.within(shapee):   #checked whether point is within the polygon or not
                t = np.array([[j],[i],[1]])
                src = np.matmul(H,t)  #calculating new co-ordinates using H matrix obtained
                srcRow = int(src[0]/src[2])
                srcCol = int(src[1]/src[2])
               
                #bound-checking
                if srcRow < 0: 
                    srcRow = 0
                if srcCol < 0: 
                    srcCol = 0
                if srcRow > source.shape[1]: 
                    srcRow = source.shape[1] - 1
                if srcCol > source.shape[0]: 
                    srcCol = source.shape[0] - 1
                
                try:
                    target[i,j] = source[srcCol,srcRow]  #replacing the target pixels with respective source pixels
                except Exception:
                    pass
    
    destination=input("Enter the Destination Path:")
    imageio.imwrite(destination,target) #writing the output image to the specified path


    
if __name__=='__main__':
    main()
    
    
    