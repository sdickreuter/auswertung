import numpy as np
from scipy.optimize import minimize,  basinhopping

import pandas as pd
import os
import matplotlib.pyplot as plt

ref_exp = np.loadtxt('/home/sei/PycharmProjects/auswertung/anni/au_yakubovsky.csv', skiprows=1)
e_exp = ref_exp[:, 0]
epsr_exp = ref_exp[:, 1] ** 2 - ref_exp[:, 2] ** 2
epsi_exp = 2 * ref_exp[:, 1] * ref_exp[:, 2]

# plt.plot(e_exp,epsr_exp)
# plt.plot(e_exp,epsi_exp)
# plt.show()

eps_inf = 1.7
omega_p = 8.7081
gamma_free = 0.0579
omega_gap = 2.6652


def eps_free(x):
    return eps_inf - omega_p ** 2 / (x ** 2 + 1j * x * gamma_free)


def eps_1(x, A1, omega_01, gamma_01):
    e1 = A1 * (-np.sqrt(omega_gap - omega_01) / (2 * (x + 1j * gamma_01) ** 2)) * np.log( 1 - ((x + 1j * gamma_01) / (omega_01)) ** 2) \
         + 2 * np.sqrt(omega_gap) / (x + 1j * gamma_01) ** 2 * np.arctanh( np.sqrt((omega_gap - omega_01) / (omega_gap))) \
         - np.sqrt(x + 1j * gamma_01 - omega_gap) / (x + 1j * gamma_01) ** 2 * np.arctanh( np.sqrt((omega_gap - omega_01) / (x + 1j * gamma_01 - omega_gap))) \
         - np.sqrt(x + 1j * gamma_01 - omega_gap) / ( x + 1j * gamma_01) ** 2 * np.arctanh( np.sqrt((omega_gap - omega_01) / (x + 1j * gamma_01 + omega_gap)))
    return e1

def eps_2(x, A2, omega_02, gamma_02):
    e2 = - A2 / ( 2 * (x + 1j * gamma_02) ** 2 ) * np.log( 1 - ( ( x + 1j * gamma_02) / ( omega_02 ) ) ** 2 )
    return e2

def eps(x,A1,A2,omega_01,omega_02,gamma_01,gamma02):
    return eps_free(x) + eps_1(x,A1,omega_01,gamma_01) + eps_2(x,A2,omega_02,gamma02)


omega = np.linspace(1,4,100)
e = eps(omega,140,45,2.575,3.65,0.25,0.4)


plt.plot(e_exp,epsr_exp)
plt.plot(e_exp,epsi_exp)
plt.plot(omega,e.real)
plt.plot(omega,e.imag)
plt.show()

x = e_exp
y = epsr_exp + 1j*epsi_exp

def err_fun(p):
    fit = eps(x, *p)
    diff = np.abs(y.real - fit.real) + np.abs(y.imag - fit.imag)
    return np.sum(diff)**2

initial_guess = np.array([140,45,2.575,3.65,0.25,0.4])
bnds = np.array([(1,500),(1,500),(2.5,3.5),(3.5,4.5),(0.1,1),(0.1,1)])

#minimizer_kwargs = {"method": "L-BFGS-B", "tol": 1e-12,'bounds':bnds}
#res = basinhopping(err_fun, initial_guess, minimizer_kwargs=minimizer_kwargs, niter=50)
res = minimize(err_fun, initial_guess, method='SLSQP',bounds = bnds, options={ 'disp': True, 'maxiter': 500})
#res = minimize(err_fun, initial_guess, method='SLSQP', options={ 'disp': True, 'maxiter': 1000})
#res = minimize(err_fun, initial_guess, method='L-BFGS-B', jac=False, options={'disp': True, 'maxiter': 500})
#res = minimize(err_fun, initial_guess, method='nelder-mead', options={'disp': True, 'maxiter': 10000,'adaptive': True})
#res = minimize(err_fun, initial_guess, method='BFGS', options={'disp': True, 'maxiter': 1000,'eps': 0.000001})
#res = minimize(err_fun, initial_guess, method='L-BFGS-B',bounds = bnds, options={ 'disp': True, 'maxiter': 500,'maxls':100})


popt = res.x
#perr = res

e = eps(omega,*popt)
plt.plot(e_exp,epsr_exp,'bx')
plt.plot(e_exp,epsi_exp,'go')
plt.plot(omega,e.real)
plt.plot(omega,e.imag)
plt.show()




