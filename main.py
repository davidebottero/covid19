#%%
'''
Created by Davide Bottero
On May, 14th 2021
'''
#import useful libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


#%%
#import Italian data

data_url = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv'
df = pd.read_csv(data_url)

#print column names and keep only useful ones
df = df[['data', 'totale_positivi', 'variazione_totale_positivi', 'nuovi_positivi', 'dimessi_guariti']]
#data until Sept, 30th
df = df.iloc[:150]
#extract useful arrays
I = df['totale_positivi'].to_numpy()
dI = df['nuovi_positivi'].to_numpy()
R = df['dimessi_guariti'].to_numpy()
dR = df['dimessi_guariti'].diff()
length = df.shape[0]
S = 60e6*np.ones((length,1))

# %%
def f(x, a, b, c, d):
    return a + b / (1. + c*np.exp(d*x))

x = np.linspace(0, length, length)
popt, pcov = opt.curve_fit(f, x, R, method="lm")
y_fit = f(x, *popt)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(x, R, 'o')
ax.plot(x, y_fit, '-', label='Recovered')
plt.show()

a = popt[0]
b = popt[1]
c = popt[2]
d = popt[3]

# %% EXPLICIT EULER

myS = np.zeros((length,1))
myI = np.zeros((length,1))
myR = np.zeros((length,1))

myS[0] = S[0]
myI[0] = I[0]
myR[0] = R[0]

alfa = a*d*(b+a)/((I[0]+R[0])*b)
beta = (alfa - d * (b+2*a)/b)/S[0]

for i in range(0, length-1):
    myS[i+1] = myS[i] - beta*I[i]*S[i]
    myI[i+1] = myI[i] + beta*S[i]*I[i] - alfa*I[i]
    myR[i+1] = myR[i] + alfa*I[i]


plt.plot(x, myI)
plt.plot(x, I, '-o')
#plt.plot(x, myR)
#plt.plot(x, R, '-o')
plt.show()

# %%
plt.plot(x, myR)
plt.plot(x, R, '-o')

# %%
plt.plot(x, myS)