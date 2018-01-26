__author__ = 'sei'

import numpy as np
from scipy.optimize import minimize, basinhopping
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as signal
import random

sns.set_context("paper")
sns.set_style("ticks")


def fftfilter(y):
    window = signal.general_gaussian(11, p=0.5, sig=10)
    filtered = signal.fftconvolve(window, y)
    filtered = (np.average(y) / np.average(filtered)) * filtered
    filtered = np.roll(filtered, -5)[:len(y)]
    # filtered = filtered[:1024]
    #peakidx = signal.find_peaks_cwt(filtered, np.arange(1, 10), noise_perc=0.01)
    #plt.plot(x, y)
    #plt.plot(x, filtered, "g")
    #plt.plot(x[peakidx], y[peakidx], "rx")
    #plt.show()
    return filtered


def gauss(x, amplitude, xo, fwhm):
    sigma = fwhm / 2.3548
    xo = float(xo)
    g = amplitude * np.exp(-np.power(x - xo, 2.) / (2 * np.power(sigma, 2.)))
    return g.ravel()


def lorentz(x, amplitude, xo, sigma):
    xo = float(xo)
    g = amplitude * np.power(sigma / 2, 2.) / (np.power(sigma / 2, 2.) + np.power(x - xo, 2.))
    return g.ravel()


# https://www.webpages.uidaho.edu/brauns/vibspect1.pdf
def asymvoigt(x, amplitude, x0, sigma, a , f):
    sigma = 2 * sigma/(1 + np.exp(a*(x-x0)) )
    g = f*lorentz(x,amplitude,x0,sigma)+(1-f)*gauss(x,amplitude,x0,sigma)
    return g.ravel()

def fit_fun(x, amp, x0, sigma,a,f):
    return asymvoigt(x, amp, x0, sigma,a,f)



# plt.plot(wl,lorentz(wl,1,500,100))
# plt.show()

def findbiggestGauss(x, y, fitwindow=20):
    window = signal.general_gaussian(11, p=0.5, sig=10)
    filtered = signal.fftconvolve(window, y)
    filtered = (np.average(y) / np.average(filtered)) * filtered
    filtered = np.roll(filtered, -5)[:len(y)]
    window = signal.general_gaussian(11, p=0.5, sig=200)
    deriv = np.gradient(filtered)
    dfiltered = signal.fftconvolve(window, deriv)
    dfiltered = (np.average(deriv) / np.average(dfiltered)) * dfiltered
    dfiltered = np.roll(dfiltered, -15)[:len(deriv)]
    zeros = np.sign(dfiltered)
    zeros[zeros == 0] = -1  # replace zeros with -1
    zeros = np.where(np.diff(zeros))[0]
    max = np.argmax(filtered[zeros])
    max = zeros[max]
    # print(x[max])
    initial_guess = (filtered[max], x[max], 10)
    minind = max - fitwindow
    if minind < 0:
        minind = 0
    maxind = max + fitwindow
    if maxind >= len(x):
        maxind = len(x) - 1
    indices = np.linspace(minind, maxind, fitwindow * 2 + 1, dtype=int)
    # popt, pcov = curve_fit(gauss, x[indices], y[indices], p0=initial_guess)
    # perr = np.sqrt(np.diag(pcov))
    def err_fun(p):
        fit = gauss(x[indices], *p)
        diff = np.abs(y[indices] - fit) ** 2
        return np.sum(diff)

    minimizer_kwargs = {"method": "SLSQP", "tol" : 1e-12}
    res = basinhopping(err_fun, initial_guess, minimizer_kwargs=minimizer_kwargs, niter=50)

    # res = minimize(err_fun, start, method='SLSQP',tol=1e-12,bounds = bnds, options={ 'disp': True, 'maxiter': 500})
    # res = minimize(err_fun, initial_guess, method='L-BFGS-B', jac=False, options={'disp': False, 'maxiter': 500})
    # res = minimize(err_fun, start, method='nelder-mead', options={'xtol': 1e-2, 'disp': True})
    popt = res.x
    fitted = gauss(x, popt[0], popt[1], popt[2])
    return popt, fitted


def findGausses(x, y, minheight, fitwindow=20):
    amp = []
    x0 = []
    sigma = []
    # y = y - min(y)
    while True:
        popt, fitted = findbiggestGauss(x, y, fitwindow)
        # print(popt)
        if popt[0] < minheight:
            break
        if popt[2] > (max(x) - min(x)):
            break
        amp.append(popt[0])
        x0.append(popt[1])
        sigma.append(popt[2])
        y = y - fitted
    return np.array(amp), np.array(x0), np.array(sigma)


def lorentzSum(x,*p):
    n = int(len(p) / 3)
    amp = p[:n]
    x0 = p[n:2 * n]
    sigma = p[2 * n:3 * n]
    res = lorentz(x, amp[0], x0[0], sigma[0])
    for i in range(len(amp) - 1):
        res += lorentz(x, amp[i + 1], x0[i + 1], sigma[i + 1])
    #res += p[-1]
    return res


def fitLorentzes(x, y, amp, x0, sigma, min_heigth, min_sigma, max_sigma):
    n = len(amp)
    def err_fun(p):
        fit = lorentzSum(x, *p)
        diff = np.abs(y - fit)
        return np.sum(diff)

    c = 0
    start = np.concatenate((amp, x0, sigma))
    n = int(len(amp))
    #upper = np.concatenate((np.repeat(max(y), n), x0 + 50, np.repeat(max_sigma, n)))
    #lower = np.concatenate((np.repeat(min(y), n), x0 - 50, np.repeat(min_sigma, n)))
    upper = np.concatenate((np.repeat(max(y), n),np.repeat(max(x), n) , np.repeat(max_sigma, n)))
    lower = np.concatenate((np.repeat(min(y), n),np.repeat(min(x), n), np.repeat(min_sigma, n)))
    bnds = []
    for i in range(len(upper)):
        bnds.append((lower[i], upper[i]))
    minimizer_kwargs = {"method": "SLSQP", "bounds" : bnds}
    #minimizer_kwargs = {"method": "nelder-mead"}
    if len(amp) > 0:
        res = basinhopping(err_fun, start, minimizer_kwargs=minimizer_kwargs, niter=500)
        #res = minimize(err_fun, start, method='SLSQP', tol=1e-12, bounds=bnds, options={'disp': False, 'maxiter': 500})
        #res = minimize(err_fun, start, method='L-BFGS-B', jac=False,bounds = bnds, options={'disp': False, 'maxiter': 500})
        # res = minimize(err_fun, start, method='nelder-mead', options={'xtol': 1e-2, 'disp': True})
        #res = minimize(err_fun, start, method='TNC', tol=1e-12, bounds=bnds, options={'disp': False, 'maxiter': 500})
        p = res.x
        amp_fit = p[:n]
        x0_fit = p[n:2 * n]
        sigma_fit = p[2 * n:3 * n]
        ind = []
        #for i in range(len(amp_fit)):
        #    if amp_fit[i] > min_heigth and sigma_fit[i] > min_sigma:
        #        ind.append(i)
        #return amp_fit[ind], x0_fit[ind], sigma_fit[ind]
        return amp_fit, x0_fit, sigma_fit
    else :
        print("No Starting Values")
        return amp, x0, sigma

def plotLorentzes(filename,x, y, amp, x0, sigma):
    cols = sns.color_palette(n_colors=len(x0) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color=cols[0])
    plt.ylabel(r'$I_{df} [a.u.]$')
    plt.xlabel(r'$\lambda [nm]$')
    plt.xlim((min(x), max(x)))
    sum = np.zeros(len(x))
    for i in range(len(x0)):
        plt.plot(x, lorentz(x, amp[i], x0[i], sigma[i]), color=cols[i + 1])
        sum += lorentz(x, amp[i], x0[i], sigma[i])
    plt.plot(x, sum, "k--")
    # plt.show()
    plt.savefig(filename, format='png')
    plt.close()


def plotGausses(filename, x, y, amp, x0, sigma):
    cols = sns.color_palette(n_colors=len(x0) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color=cols[0])
    plt.ylabel(r'$I_{df} [a.u.]$')
    plt.xlabel(r'$\lambda [nm]$')
    plt.xlim((min(x), max(x)))
    sum = np.zeros(len(x))
    for i in range(len(x0)):
        plt.plot(x, gauss(x, amp[i], x0[i], sigma[i]), color=cols[i + 1])
        sum += gauss(x, amp[i], x0[i], sigma[i])
    plt.plot(x, sum, "k--")
    # plt.show()
    plt.savefig(filename, format='png')
    plt.close()


def fitLorentzes2(x, y, min_heigth, min_sigma, max_sigma, max_iter=10):
    amp = np.array([1])
    x0 = np.array([np.mean(x)])
    sigma = np.array([1])
    n = 1
    def err_fun(p):
        fit = lorentzSum(x, *p)
        diff = np.abs(y - fit) ** 2
        return np.sum(diff)
    for i in range(max_iter):
        start = np.concatenate((amp, x0, sigma))
        upper = np.concatenate((np.repeat(max(y) * 2, n), np.repeat(max(x), n), np.repeat(max_sigma, n)))
        lower = np.concatenate((np.repeat(0, n), np.repeat(min(x), n), np.repeat(0, n)))
        bnds = []
        for i in range(len(upper)):
            bnds.append((lower[i], upper[i]))
        #minimizer_kwargs = {"method": "SLSQP", "tol" : 1e-12, "bounds" : bnds}
        #res = basinhopping(err_fun, start, minimizer_kwargs=minimizer_kwargs, niter=10)
        res = minimize(err_fun, start, method='SLSQP', tol=1e-12, bounds=bnds, options={'disp': False, 'maxiter': 500})
        # res = minimize(err_fun, start, method='L-BFGS-B', jac=False, bounds=bnds, options={'disp': False, 'maxiter': 500})
        # res = minimize(err_fun, start, method='nelder-mead', options={'xtol': 1e-2, 'disp': True})
        p = res.x
        amp_fit = p[:n]
        x0_fit = p[n:2 * n]
        sigma_fit = p[2 * n:3 * n]
        #for i in range(len(amp_fit)):
        #    if amp_fit[i] < min_heigth or sigma_fit[i] < min_sigma:
        #        return amp_fit, x0_fit, sigma_fit
        amp = np.concatenate((amp_fit, [np.mean(amp_fit)]))
        x0 = np.concatenate((x0_fit, [np.random.randint(min(x),max(x))]))
        sigma = np.concatenate((sigma_fit, [sigma_fit[len(sigma_fit)-1]]))
        #print(res.fun)
        # ind = []
        n += 1
    return amp_fit, x0_fit, sigma_fit




def fitLorentzes3(x, y, amp, x0, sigma, min_heigth, min_sigma, max_sigma):
    amp_buf = np.array([amp[0]])
    x0_buf = np.array([x0[0]])
    sigma_buf = np.array([sigma[0]])
    # def err_fun(p):
    #     fit = lorentzSum(x, *p)
    #     filt = fftfilter(y)
    #     diff = np.power(filt - fit, 2.)
    #     err = np.sum(diff)
    #     n = int(len(p) / 3)
    #     #amp = p[:n]
    #     #x0 = p[n:2 * n]
    #     sigma = p[2 * n:3 * n]
    #     sig_err = 0
    #     for i in range(len(sigma)):
    #         if sigma[i] > max_sigma or sigma[i] < min_sigma:
    #             sig_err += (sigma[i]-max_sigma)**2
    #     #print(sig_err)
    #     return err+sig_err*0.001
    def err_fun(p):
        fit = lorentzSum(x, *p)
        diff = np.abs(y - fit) ** 2
        return np.sum(diff)
    last_fun = np.sum(y)
    n = 1
    #for i in range(len(amp)-1):
    for i in range(len(amp)):
        start = np.concatenate((amp_buf, x0_buf, sigma_buf))
        upper = np.concatenate((np.repeat(max(y), n), np.repeat(max(x), n), np.repeat(max_sigma, n)))
        lower = np.concatenate((np.repeat(0, n), np.repeat(min(x), n), np.repeat(min_sigma, n)))
        #upper = np.concatenate((np.repeat(max(y) * 2, n), np.repeat(max(x), n), np.repeat(1000, n)))
        #lower = np.concatenate((np.repeat(0, n), np.repeat(min(x), n), np.repeat(0, n)))
        #upper = np.concatenate((np.repeat(max(y), i+1), x0_buf + 30, np.repeat(1000, i+1)))
        #lower = np.concatenate((np.repeat(0, i+1), x0_buf - 30, np.repeat(0, i+1)))
        bnds = []
        for j in range(len(upper)):
            bnds.append((lower[j], upper[j]))
        minimizer_kwargs = {"method": "L-BFGS-B", "tol" : 1e-12, "bounds" : bnds}
        res = basinhopping(err_fun, start, minimizer_kwargs=minimizer_kwargs, niter=10)
        #res = minimize(err_fun, start, method='SLSQP', tol=1e-12, bounds=bnds, options={'disp': False, 'maxiter': 500})
        #res = minimize(err_fun, start, method='L-BFGS-B', jac=False, bounds=bnds, options={'disp': False, 'maxiter': 500})
        #res = minimize(err_fun, start, method='nelder-mead', options={'xtol': 1e-2, 'disp': True})
        p = res.x
        amp_fit = p[:n]
        x0_fit = p[n:2 * n]
        sigma_fit = p[2 * n:3 * n]
        #for i in range(len(amp_fit)):
        #    if amp_fit[i] < min_heigth or sigma_fit[i] < min_sigma:
        #        return amp_fit, x0_fit, sigma_fit
        amp_buf = np.concatenate((amp_fit, [amp[n]]))
        x0_buf = np.concatenate((x0_fit, [x0[n]]))
        sigma_buf = np.concatenate((sigma_fit, [sigma[n]]))
        # ind = []
        if res.fun > last_fun:
            break
        else:
            last_fun = res.fun
        n += 1
    return amp_fit, x0_fit, sigma_fit



def fitLorentzes_iter(x, y, min_heigth, min_sigma, max_sigma, iter):
    def err_fun(p):
        fit = lorentzSum(x, *p)
        diff = np.abs(y - fit)
        return np.sum(diff)
        #return np.sum(diff)*(len(p)/30)
        #return (np.mean(diff)+np.max(diff)/2)
        #return (np.mean(diff)+np.max(diff))/(len(p)/3)
        #return np.max(diff)


    #y = signal.savgol_filter(y, 51, 1, mode='interp')
    #plt.plot(x,y)
    #plt.show()

    #iter = 10
    fun = np.inf
    n=1
    p = None
    while True:
        start = np.concatenate((np.repeat(max(y) / 5, n), np.linspace(np.min(x)+20,np.max(x)-20,n), np.repeat(50, n)))
        #start = np.append(start,0)
        #start = np.concatenate((np.repeat(max(y) / 5, n), np.random.randint(np.min(x)+20,np.max(x)-20,1), np.repeat(50, n)))

        upper = np.concatenate((np.repeat(max(y), n), np.repeat(np.max(x), n), np.repeat(max_sigma, n)))
        #upper = np.append(upper, 1)
        lower = np.concatenate((np.repeat(min_heigth, n), np.repeat(np.min(x), n), np.repeat(min_sigma, n)))
        #lower = np.append(lower, 0)
        bnds = []
        for i in range(len(upper)):
            bnds.append((lower[i], upper[i]))
        minimizer_kwargs = {"method": 'L-BFGS-B', "bounds" : bnds}
        #minimizer_kwargs = {"method": 'L-BFGS-B'}
        res = basinhopping(err_fun, start, minimizer_kwargs=minimizer_kwargs, niter=500)
        #res = minimize(err_fun, start, method='SLSQP', tol=1e-20, options={'disp': False, 'maxiter': 500})
        #res = minimize(err_fun, start, method='SLSQP', tol=1e-20, bounds=bnds, options={'disp': False, 'maxiter': 500})
        #res = minimize(err_fun, start, method='Powell', bounds=bnds, options={'disp': False})
        #res = minimize(err_fun, start, method='Powell', tol=1e-20, bounds=bnds, options={'disp': False})
        #res = minimize(err_fun, start, method='L-BFGS-B', jac=False,bounds = bnds, options={'disp': True, 'maxiter': 500})
        #res = minimize(err_fun, start, method='nelder-mead', options={'xtol': 1e-6, 'disp': True})
        #res = minimize(err_fun, start, method='TNC', tol=1e-15, bounds=bnds, options={'disp': False, 'maxiter': 500})
        last_p = p
        p = res.x
        last_fun = fun
        fun = res.fun
        n += 1
        print(fun)
        #if fun < 0.00002:
        #    #print(fun)
        #    break
        if (fun > last_fun):
            p = last_p
            n -= 1
            break

        if n > iter:
            #n -= 1
            break

        #print(p)

    amp_fit = p[:(n-1)]
    x0_fit = p[(n-1):(2 * (n-1))]
    sigma_fit = p[(2 * (n-1)):(3 * (n-1))]
    #print(p[-1])
    #print([amp_fit,x0_fit,sigma_fit])
    return amp_fit, x0_fit, sigma_fit




def findbiggestLorentz(x, y, fitwindow):
    filtered = signal.savgol_filter(y, 61, 1, mode='interp')
    deriv = np.gradient(filtered)
    dfiltered = signal.savgol_filter(deriv, 91, 1, mode='interp')
    dderiv = np.gradient(dfiltered)
    ddfiltered = signal.savgol_filter(dderiv, 91, 1, mode='interp')
    ddderiv = np.gradient(ddfiltered)
    dddfiltered = signal.savgol_filter(ddderiv, 91, 1, mode='interp')
    dddderiv = np.gradient(dddfiltered)
    ddddfiltered = signal.savgol_filter(dddderiv, 91, 1, mode='interp')
    filtered = ddddfiltered
    filtered = filtered - min(filtered)

    #plt.plot(x,filtered/np.max(filtered))
    #plt.plot(x,dfiltered/np.max(dfiltered))
    #plt.plot(x,ddddfiltered/np.max(ddddfiltered))
    #plt.axhline(y=0, color = 'k')
    #plt.show()

    #zeros = np.sign(dfiltered)
    #zeros[zeros == 0] = -1  # replace zeros with -1
    #zeros = np.where(np.diff(zeros))[0]
    #mask = ddfiltered[zeros] < 0
    #zeros = zeros[mask]
    #max = np.argmax(filtered[zeros])
    #max = zeros[max]

    max = np.argmax(filtered)

    initial_guess = (filtered[max], x[max], 100)
    minind = max - fitwindow
    if minind < 0:
        minind = 0
    maxind = max + fitwindow
    if maxind >= len(x):
        maxind = len(x) - 1
    indices = np.linspace(minind, maxind,maxind-minind , dtype=int)
    # popt, pcov = curve_fit(gauss, x[indices], y[indices], p0=initial_guess)
    # perr = np.sqrt(np.diag(pcov))
    #print(indices)
    def err_fun(p):
        amp = p[0]
        x0 = p[1]
        fwhm = p[2]
        fit = gauss(x[indices],amp,x0,fwhm)
        diff = np.abs(filtered[indices] - fit)
        return np.sum(diff)

    bnds = ( (np.min(filtered),np.max(filtered)),(x[minind],x[maxind]),(1,500) )
    minimizer_kwargs = {"method": "SLSQP", "bounds" : bnds}
    res = basinhopping(err_fun, initial_guess, minimizer_kwargs=minimizer_kwargs, niter=50)
    #res = minimize(err_fun, initial_guess, method='SLSQP', options={ 'disp': True, 'maxiter': 500})
    #res = minimize(err_fun, initial_guess, method='L-BFGS-B', jac=False, options={'disp': True, 'maxiter': 500})
    ## res = minimize(err_fun, start, method='nelder-mead', options={'xtol': 1e-2, 'disp': True})
    popt = res.x
    #popt = [y[max],x[max],100]
    popt[0] = y[max]
    fitted = gauss(x, *popt)
    #plt.plot(x,filtered)
    #plt.plot(x,fitted)
    plt.show()

    return popt, fitted

def findLorentzes(x, y, minheight, fitwindow):
    amp = []
    x0 = []
    sigma = []
    # y = y - min(y)
    while True:
        popt, fitted = findbiggestLorentz(x, y, fitwindow)
        # print(popt)
        if popt[0] < minheight:
            break
        if popt[2] > (max(x) - min(x)):
            break
        if popt[1] > max(x):
            break
        if popt[1] < min(x):
            break
        amp.append(popt[0])
        #ind = np.min(np.where(x > popt[1]))
        #amp.append(y[ind])
        x0.append(popt[1])
        sigma.append(popt[2])
        y = y - fitted
    return np.array(amp), np.array(x0), np.array(sigma)
