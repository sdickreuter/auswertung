__author__ = 'sei'

from gauss_detect import *
from scipy.optimize import basinhopping, minimize

# from peakdetect import peakdetect

path = '/home/sei/Spektren/2C1/'
# sample = '2C1_150hex_B3'
# sample = '2C1_150hept_B2'
# sample = '2C1_200hex_A2'
# sample = '2C1_200hex_B1'
sample = '2C1_75hept_B2'
# sample = '2C1_100hex_C2'

savedir = path + sample + '/'

# grid dimensions
nx = 7
ny = 7

# try:
#    os.mkdir(path+savedir)
# except:
#    pass

wl, lamp = np.loadtxt(open(savedir + "lamp.csv", "rb"), delimiter=",", skiprows=8, unpack=True)
wl, dark = np.loadtxt(open(savedir + "dark.csv", "rb"), delimiter=",", skiprows=8, unpack=True)
wl, bg = np.loadtxt(open(savedir + "background.csv", "rb"), delimiter=",", skiprows=8, unpack=True)
wl, spec = np.loadtxt(open(savedir + "specs/A3_corr.csv", "rb"), delimiter=",", skiprows=12, unpack=True)

wl = wl[250:950]
spec = spec[250:950]

# from gauss_detect import findGausses
# spec = spec - np.mean(spec[0:15])

amp, x0, sigma = findGausses(wl, spec, 0.01, 30)
cols = sns.color_palette(n_colors=len(x0) + 1)
plt.plot(wl, spec, color=cols[0])
sum = np.zeros(len(wl))
for i in range(len(x0)):
    plt.plot(wl, gauss(wl, amp[i], x0[i], sigma[i]), color=cols[i + 1])
    sum += gauss(wl, amp[i], x0[i], sigma[i])

plt.plot(wl, sum, "k--")
plt.show()


def lorentzSum(x, *p):
    # p = p[0]
    n = int(len(p) / 3)
    amp = p[:n]
    x0 = p[n:2 * n]
    sigma = p[2 * n:3 * n]
    res = lorentz(x, amp[0], x0[0], sigma[0])
    for i in range(len(amp) - 1):
        res += lorentz(x, amp[i + 1], x0[i + 1], sigma[i + 1])
    return res


initial_guess = tuple(np.concatenate((amp, x0, sigma)))
popt, pcov = curve_fit(lorentzSum, wl, spec, p0=initial_guess)
perr = np.sqrt(np.diag(pcov))


def err_fun(p):
    fit = lorentzSum(wl, *p)
    diff = np.abs(spec - fit) ** 2
    return np.sum(diff)


start = np.concatenate((amp, x0, sigma))
n = int(len(amp))

upper = np.concatenate((np.repeat(max(spec) * 100, n), x0 + 20, np.repeat(max(sigma), n)))
lower = np.concatenate((np.repeat(0, n), x0 - 20, np.repeat(1, n)))

bnds = []
for i in range(len(upper)):
    bnds.append((lower[i], upper[i]))

# res = minimize(err_fun, start, method='SLSQP',tol=1e-12,bounds = bnds, options={ 'disp': True, 'maxiter': 500})
res = minimize(err_fun, start, method='L-BFGS-B', jac=False, bounds=bnds, options={'disp': False, 'maxiter': 500})
# res = minimize(err_fun, start, method='nelder-mead', options={'xtol': 1e-2, 'disp': True})
p = res.x
amp_fit = p[:n]
x0_fit = p[n:2 * n]
sigma_fit = p[2 * n:3 * n]

cols = sns.color_palette(n_colors=len(x0) + 1)
plt.plot(wl, spec, color=cols[0])
plt.ylabel(r'$I_{df} [a.u.]$')
plt.xlabel(r'$\lambda [nm]$')
plt.xlim((min(wl), max(wl)))
sum = np.zeros(len(wl))
for i in range(len(x0)):
    plt.plot(wl, lorentz(wl, amp_fit[i], x0_fit[i], sigma_fit[i]), color=cols[i + 1])
    sum += lorentz(wl, amp_fit[i], x0_fit[i], sigma_fit[i])

plt.plot(wl, sum, "k--")
plt.show()

amp, x0, sigma = findGausses(wl, spec, 0.01, 30)
amp, x0, sigma = fitLorentzes(wl, spec, amp, x0, sigma)
plotLorentzes(wl, spec, amp, x0, sigma)


def func(x, p): return p[0] + p[1] * x


def func2(*args):
    return func(args[0], args[1:])


popt, pcov = curve_fit(func2, np.arange(10), np.arange(10), p0=(0, 0))
print(popt, pcov)


def gauss(x, amplitude, xo, fwhm):
    sigma = fwhm / 2.3548
    xo = float(xo)
    g = amplitude * np.exp(-np.power(x - xo, 2.) / (2 * np.power(sigma, 2.)))
    return g.ravel()


def lorentz(x, amplitude, xo, fwhm):
    sigma = fwhm / 2.3548
    xo = float(xo)
    g = 2 * np.pi * amplitude * sigma / (np.power(sigma, 2.) + np.power(x - xo, 2.))
    return g.ravel()


window = signal.general_gaussian(11, p=0.5, sig=10)
filtered = signal.fftconvolve(window, spec)
filtered = (np.average(spec) / np.average(filtered)) * filtered
filtered = np.roll(filtered, -5)[:len(spec)]
# filtered = filtered[:1024]
peakidx = signal.find_peaks_cwt(filtered, np.arange(1, 10), noise_perc=0.01)
plt.plot(wl, spec)
plt.plot(wl, filtered, "g")
plt.plot(wl[peakidx], spec[peakidx], "rx")
plt.show()

window = signal.general_gaussian(31, p=0.5, sig=200)
deriv = np.gradient(filtered)
dfiltered = signal.fftconvolve(window, deriv)
dfiltered = (np.average(deriv) / np.average(dfiltered)) * dfiltered
dfiltered = np.roll(dfiltered, -15)[:len(deriv)]
# spl = UnivariateSpline(wl, deriv)
# spl.set_smoothing_factor(30)
# dfiltered = spl(wl)
dderiv = np.gradient(dfiltered)
buf = (spec / max(spec)) * max(dfiltered)
plt.plot(wl, deriv)
plt.plot(wl, dderiv, "y")
plt.hlines(0, min(wl), max(wl))
plt.plot(wl, dfiltered, "g")
plt.plot(wl, buf, "r")
plt.show()

zeros = np.sign(dfiltered)
zeros[zeros == 0] = -1  # replace zeros with -1
zeros = np.where(np.diff(zeros))[0]
print(wl[zeros])
max = np.argmax(filtered[zeros])
max = zeros[max]
print(wl[max])

# gauss(x, amplitude, xo, fwhm, offset)
initial_guess = (filtered[max], wl[max], 10)
indices = np.linspace(max - 20, max + 20, 41, dtype=int)
popt, pcov = curve_fit(lorentz, wl[indices], spec[indices], p0=initial_guess)
perr = np.sqrt(np.diag(pcov))
fitted = lorentz(wl, popt[0], popt[1], popt[2])

plt.plot(wl, filtered)
plt.plot(wl, fitted, "g")
plt.plot(wl, filtered - fitted, "r")
plt.show()


def findbiggestGauss(x, y, fitwindow=20):
    window = signal.general_gaussian(11, p=0.5, sig=10)
    filtered = signal.fftconvolve(window, y)
    filtered = (np.average(y) / np.average(filtered)) * filtered
    filtered = np.roll(filtered, -5)[:len(y)]
    # filtered = filtered[:1024]
    window = signal.general_gaussian(31, p=0.5, sig=200)
    deriv = np.gradient(filtered)
    dfiltered = signal.fftconvolve(window, deriv)
    dfiltered = (np.average(deriv) / np.average(dfiltered)) * dfiltered
    dfiltered = np.roll(dfiltered, -15)[:len(deriv)]
    # spl = UnivariateSpline(wl, deriv)
    # spl.set_smoothing_factor(30)
    # dfiltered = spl(wl)
    # buf = (spec / max(spec) )*max(dfiltered)
    # plt.plot(wl, deriv)
    # plt.plot(wl, dderiv,"y")
    # plt.hlines(0,min(wl),max(wl))
    # plt.plot(wl, dfiltered,"g")
    # plt.plot(wl, buf,"r")
    # plt.show()
    zeros = np.sign(dfiltered)
    zeros[zeros == 0] = -1  # replace zeros with -1
    zeros = np.where(np.diff(zeros))[0]
    # print(wl[zeros])
    max = np.argmax(filtered[zeros])
    max = zeros[max]
    # print(wl[max])
    # gauss(x, amplitude, xo, fwhm, offset)
    initial_guess = (filtered[max], x[max], 10)
    indices = np.linspace(max - fitwindow, max + fitwindow, fitwindow * 2 + 1, dtype=int)
    popt, pcov = curve_fit(gauss, x[indices], y[indices], p0=initial_guess)
    perr = np.sqrt(np.diag(pcov))
    fitted = gauss(wl, popt[0], popt[1], popt[2])
    return popt, perr, fitted


popt1, perr1, fitted = findbiggestGauss(wl, spec)
plt.plot(wl, spec)
plt.plot(wl, fitted, "g")
plt.plot(wl, spec - fitted, "r")
plt.show()

buf = spec - fitted
popt2, perr2, fitted = findbiggestGauss(wl, buf)
plt.plot(wl, buf)
plt.plot(wl, fitted, "g")
plt.plot(wl, buf - fitted, "r")
plt.show()

buf = buf - fitted
popt3, perr3, fitted = findbiggestGauss(wl, buf)
plt.plot(wl, buf)
plt.plot(wl, fitted, "g")
plt.plot(wl, buf - fitted, "r")
plt.show()

buf = buf - fitted
popt4, perr4, fitted = findbiggestGauss(wl, buf)
plt.plot(wl, buf)
plt.plot(wl, fitted, "g")
plt.plot(wl, buf - fitted, "r")
plt.show()

buf = buf - fitted
popt5, perr5, fitted = findbiggestGauss(wl, buf)
plt.plot(wl, buf)
plt.plot(wl, fitted, "g")
plt.plot(wl, buf - fitted, "r")
plt.show()

buf = buf - fitted
popt6, perr6, fitted = findbiggestGauss(wl, buf)
plt.plot(wl, buf)
plt.plot(wl, fitted, "g")
plt.plot(wl, buf - fitted, "r")
plt.show()

x = wl
y = spec
max_sigma = 500
min_sigma = 30
max_iter = 5

amp = np.array([1])
x0 = np.array([np.random.randint(min(wl),max(wl))])
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
    #res = minimize(err_fun, start, method='SLSQP', tol=1e-12, bounds=bnds, options={'disp': False, 'maxiter': 500})
    #res = minimize(err_fun, start, method='TNC', tol=1e-12, bounds=bnds, options={'disp': False, 'maxiter': 500})
    #res = minimize(err_fun, start, method='L-BFGS-B', jac=False, bounds=bnds, options={'disp': False, 'maxiter': 500})
    #res = minimize(err_fun, start, method='nelder-mead', options={'xtol': 1e-2, 'disp': True})
    minimizer_kwargs = {"method": "SLSQP", "tol" : 1e-12, "bounds" : bnds}
    res = basinhopping(err_fun, start, minimizer_kwargs=minimizer_kwargs, niter=50)
    p = res.x
    amp_fit = p[:n]
    x0_fit = p[n:2 * n]
    sigma_fit = p[2 * n:3 * n]
    #for i in range(len(amp_fit)):
    #    if amp_fit[i] < min_heigth or sigma_fit[i] < min_sigma:
    #        return amp_fit, x0_fit, sigma_fit
    amp = np.concatenate((amp_fit, [np.mean(amp_fit)]))
    x0 = np.concatenate((x0_fit, [np.random.randint(min(wl),max(wl))]))
    sigma = np.concatenate((sigma_fit, [sigma_fit[len(sigma_fit)-1]]))
    n += 1
    #print(res.fun)

plotLorentzes("lorentz.png", x, y, amp_fit, x0_fit, sigma_fit)
