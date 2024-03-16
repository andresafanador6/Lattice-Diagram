# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:06:59 2024

@author: ASUS
"""
import numpy as np
def coef(impe1,impe2,impe3,impe4):
    alfa12=(impe2-impe1)/(impe1+impe2)
    beta12=2*impe2/(impe1+impe2)
    alfa21=(impe1-impe2)/(impe1+impe2)
    beta21=2*impe1/(impe1+impe2)
    alfa23=(impe3-impe2)/(impe3+impe2)
    alfa32=(impe2-impe3)/(impe3+impe2)
    beta23=2*impe3/(impe2+impe3)
    beta32=2*impe2/(impe2+impe3)
    beta34=2*impe4/(impe4+impe3)
    alfa34=(impe4-impe3)/(impe3+impe4)
    beta43=2*impe3/(impe4+impe3)
    alfa43=(impe3-impe4)/(impe4+impe4)
    return alfa12,beta12,alfa21,beta21,alfa23,alfa32,beta23,beta32,beta34,alfa34,beta43,alfa43

def collision(array,arraycollision,col1,alfaAB,alfaBA,alfaBC,alfaCB,betaAB,betaBA,betaBC,betaCB
              ,X,V,tmax,t1,val1):
    "Calculate colision of 1 impact"
    
    val=val1
    i=t1
    array[i,col1]=val1
    
    while i<tmax:
        
        if i+(X/V)<tmax:
            i=i+(X/V)
            
        else :
            break
        
        
        array[int(i),int(col1+2)]=array[int(i),int(col1+2)]+val*betaBC
        arraycollision[int(i),int(col1+2)]=arraycollision[int(i),int(col1+2)]+val*betaBC
        val=val*alfaBC
        array[int(i),int(col1+1)]=val+array[int(i),int(col1+1)]
        
        if i+(X/V)<tmax:
            i=i+(X/V)
        else :
            break
        array[int(i),int(col1-1)]=val*betaBA+array[int(i),int(col1-1)]
        val=val*alfaBA
        arraycollision[int(i),int(col1-1)]=arraycollision[int(i),int(col1-1)]+val*alfaBA
        array[int(i),int(col1)]=val+ array[int(i),int(col1)]
    
    array[:,0]=0
    arraycollision[:,0]=0
    
    return array,arraycollision

def collision2(array,arraycollision,
               col1,alfa23,alfa32,alfa34,alfa43,beta23,beta32,beta34,beta43
              ,X,V,tmax,t1,val1):
    "Calculate colision of 1 impact"
    
    val=val1
    i=t1
    array[i,col1]=val1
    
    while i<tmax:
        
        if i+(X/V)<tmax:
            i=i+(X/V)
            
        else :
            break
        
        
        array[int(i),int(col1+2)]=array[int(i),int(col1+2)]+val*beta34
         
        val=val*alfa34
        array[int(i),int(col1+1)]=val+array[int(i),int(col1+1)]
        
        if i+(X/V)<tmax:
            i=i+(X/V)
        else :
            break
        
        array[int(i),int(col1-1)]=val*beta32+array[int(i),int(col1-1)]
        
        
        arraycollision[int(i),int(col1-1)]=arraycollision[int(i),int(col1-1)]+val*beta32
        val=val*alfa32
        array[int(i),int(col1)]=val+ array[int(i),int(col1)]
        
    array[:,0]=0
    arraycollision[:,0]=0
    
    return array,arraycollision
def collision3(array,arraycollision,
               col1,alfaAB,alfaBA,alfaBC,alfaCB,betaAB,betaBA,betaBC,betaCB
              ,X,V,tmax,t1,val1):
    "Calculate colision of 1 impact"
    
    val=val1
    i=t1 
    array[i,col1+1]=val1
    
    while i<tmax:
        
        if i+(X/V)<tmax:
            i=i+(X/V)
        else :
            break
        array[int(i),int(col1-1)]=val*betaBA+array[int(i),int(col1-1)]
        val=val*alfaBA
        
        
        arraycollision[int(i),int(col1-1)]=arraycollision[int(i),int(col1-1)]+val*alfaBA
        array[int(i),int(col1)]=val+ array[int(i),int(col1)]
        
        if i+(X/V)<tmax:
            i=i+(X/V)
            
        else :
            break
        
        
        array[int(i),int(col1+2)]=array[int(i),int(col1+2)]+val*betaBC
        arraycollision[int(i),int(col1+2)]=arraycollision[int(i),int(col1+2)]+val*betaBC
        val=val*alfaBC
        array[int(i),int(col1+1)]=val+array[int(i),int(col1+1)]
        
        
    array[:,0]=0
    arraycollision[:,0]=0
    
    return array,arraycollision
def calculus(x2,x3,v2,v3,ohm1,ohm2,ohm3,ohm4,tmax):
    array=np.zeros((tmax,6))
    arraycollision1=np.zeros((tmax,6))
    "Calculate colision of 1 column"
    
    alfa12,beta12,alfa21,beta21,alfa23,alfa32,beta23,beta32,beta34,alfa34,beta43,alfa43=coef(ohm1,ohm2,ohm3,ohm4)
    
    array,arraycollision1=collision(array,arraycollision1,1,
                           alfa12,alfa21,alfa23,alfa32,beta12,beta21,beta23,beta32,
                           x2,v2,tmax,0,1)
    
    for i in range(np.shape(arraycollision1)[0]):
        for j in range(np.shape(arraycollision1)[1]):
            if arraycollision1[i,j]!=float(0):
                
                if j==2:
                    array,arraycollision1=collision3(array,arraycollision1,1,
                                           alfa12,alfa21,alfa23,alfa32,beta12,beta21,beta23,beta32,
                                           x2,v2,tmax,i,arraycollision1[i][j])
                    
                
                if j==3:
                    
                    array,arraycollision1=collision2(array,arraycollision1,j,
                                           alfa23,alfa32,alfa34,alfa43,beta23,beta32,beta34,beta43,  
                                           x3,v3,tmax,i,arraycollision1[i][j])
    """
    x,y=collision2(np.zeros((tmax,6)),np.zeros((tmax,6)),1,
                           alfa12,alfa21,alfa23,alfa32,beta12,beta21,beta23,beta32,
                           x3,v3,tmax,0,1)
    """
    
    return array

mat=calculus(3000,100,300,100,0,40,400,1e100,41)
print("Los valores son")
print(mat)
#print(mat[12:45,:])
#print(mat)
x1=mat[:,1]
x2=mat[:,2]
x3=mat[:,3]
x4=mat[:,4]
x5=mat[:,5]

y1=np.nonzero(mat[:,1].reshape(1, -1)[0])
#y1=np.full((1, np.shape(y1)[0]), np.shape(mat[:,1])[0])[0]-y1

x1=x1[x1!= 0].reshape(1, -1)

y2=np.nonzero(mat[:,2].reshape(1, -1)[0])
#y2=np.full((1, np.shape(y1)[0]), np.shape(mat[:,1])[0])[0]-y2
x2=x2[x2!= 0].reshape(1, -1)

y3=np.nonzero(mat[:,3].reshape(1, -1)[0])
#y3=np.full((1, np.shape(y1)[0]), np.shape(mat[:,1])[0])[0]-y3
x3=x3[x3!= 0].reshape(1, -1)

y4=np.nonzero(mat[:,4].reshape(1, -1)[0])
#y4=np.full((1, np.shape(y1)[0]), np.shape(mat[:,1])[0])[0]-y4
x4=x4[x4!= 0].reshape(1, -1)

y5=np.nonzero(mat[:,5].reshape(1, -1)[0])
#y5=np.full((1, np.shape(y1)[0]), np.shape(mat[:,1])[0])[0]-y5
x5=x5[x5!= 0].reshape(1, -1)






from matplotlib import pyplot as plt 
plt.subplot(1, 2, 1)
plt.plot(list(np.full(len(list(y1)[0]), 1.25)),list(y1)[0],'*',
         list(np.full(len(list(y2)[0]), 1.75)),list(y2)[0],'*',
        list(np.full(len(list(y3)[0]), 2.25)),list(y3)[0],'*',
         list(np.full(len(list(y4)[0]), 2.75)),list(y4)[0],'*',
         list(np.full(len(list(y5)[0]), 3.25)),list(y5)[0],'*')

plt.axvline(1);

plt.axvline(2);

plt.axvline(3);
plt.xlabel('Medios')
plt.ylabel('Tiempo')
plt.title('Lattice')

plt.legend()  # Mostrar leyenda
plt.gca().invert_yaxis()

# Mostrar el gráfico
plt.grid(True)
plt.show()
plt.subplot(1, 2, 2)
plt.title("Grafica de salida")
x5acum=np.cumsum(x5)
plt.plot(list(y5)[0],list(x5)[0],label="Salida")
plt.plot(list(y5)[0],list(x5acum),label="Salida acumulativa")
plt.xlabel("Tiempo")
plt.legend() 
plt.grid()


