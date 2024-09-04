#%%
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble" : r'\usepackage{siunitx}',
    "font.family": "Computer Modern Serif"
})
def Zeff(R, L, Cp, Cc, w):
    return (1/(1j*w*Cc) + 1/(1/(1j*w*L) + 1j*w*Cp + 1/R))

Z0 = 50.0 # Ohm
Cp = 500e-15 # F
Cc = 100e-15 # F
L = 41.67e-9 # H

Z = lambda w, R, L: Zeff(R, L, Cp, Cc, w)

Gamma = lambda w, R, L=L: (Z(w, R, L)-Z0)/(Z(w, R, L)+Z0)

ROn = 50e3 # Ohm
ROff = 100e3 # Ohm
RUnderCoupled = 25e3 # Ohm
R = np.linspace(0, ROff*100, 1000)

w0 = np.sqrt(1/(L*(Cc+Cp)))

# w with a mesh denser around w0
dw = 0.03
nw = 100001
w = np.concatenate((
    np.log10(np.linspace(10**(1-dw), 10, nw//2+1, endpoint=False)),
    np.geomspace(1, 1+dw, nw//2)
))*w0


frec = w/(2*np.pi*1e9)
plt.plot(frec, np.abs(Gamma(w, ROn)))
plt.plot(frec, np.abs(Gamma(w, ROff)))
plt.plot(frec, np.abs(Gamma(w, RUnderCoupled)))
plt.plot(frec, np.abs(Gamma(w, ROn, L=0.99*L)))
plt.xticks(list(np.around(plt.xticks()[0], 2)) + [frec[np.argmin(np.abs(Gamma(w, ROn)))]])
plt.xlabel(r'\(\omega (\unit{\giga\hertz})\)')
plt.ylabel(r'\(|\Gamma|\)')
plt.ylim(0, 1)
plt.gca().tick_params(axis='x', rotation=45)
plt.legend([r'\(Z (R = 50\unit{\kilo\ohm}\text{, }L = L_0)\)',
            r'\(Z_{\text{over}} (R = 100\unit{\kilo\ohm}\text{, }L = L_0)\)',
            r'\(Z_{\text{under}} (R = 25\unit{\kilo\ohm}\text{, }L = L_0)\)',
            r'\(Z_{\lambda} (R = 50\unit{\kilo\ohm}\text{, }L = \lambda L_0)\)'])
# plt.title('Reflection coefficient modulus vs frequency')
# plt.savefig('code/img/RLCParallel/ReflecCoeff.png', bbox_inches='tight', dpi=600)
plt.savefig('code/img/RLCParallel/ReflecCoeff.svg')
plt.show()

# Graph the phase of the reflection coefficient in the interval -π to π

plt.yticks(np.arange(-2*np.pi, 2*np.pi+np.pi/2, step=(np.pi/2)), [r'\(-2\pi\)',
                                                                  r'\(-3\pi/2\)',
                                                                  r'\(-\pi\)',
                                                                  r'\(-\pi/2\)',
                                                                  r'\(0\)',
                                                                  r'\(\pi/2\)',
                                                                  r'\(\pi\)',
                                                                  r'\(3\pi/2\)',
                                                                  r'\(2\pi\)'])

theta = np.angle(Gamma(w, ROn))
plt.plot(frec, theta)

thetaP = np.angle(Gamma(w, ROff))
# This is done to plot the discontinutity
pos = np.where(np.abs(np.diff(thetaP)) >= np.pi/2)[0]+1
x = np.insert(frec, pos, np.nan)
y = np.insert(thetaP, pos, np.nan)
plt.plot(x, y)

thetaPP = np.angle(Gamma(w, RUnderCoupled))
plt.plot(frec, thetaPP)

thetaL = np.angle(Gamma(w, ROn, L=0.99*L))
pos = np.where(np.abs(np.diff(thetaL)) >= np.pi/2)[0]+1
x = np.insert(frec, pos, np.nan)
y = np.insert(thetaL, pos, np.nan)
plt.plot(x, y)

plt.legend([r'\(Z (R = 50\unit{\kilo\ohm}\text{, }L = L_0)\)',
            r'\(Z_{\text{over}} (R = 100\unit{\kilo\ohm}\text{, }L = L_0)\)',
            r'\(Z_{\text{under}} (R = 25\unit{\kilo\ohm}\text{, }L = L_0)\)',
            r'\(Z_{\lambda} (R = 50\unit{\kilo\ohm}\text{, }L = \lambda L_0)\)'])
plt.xlabel(r'\(\omega (\unit{\kilo\hertz})\)')
plt.ylabel(r'\(\phi (\unit{\radian})\)')
plt.ylim(-np.pi, np.pi)
# plt.savefig('code/img/RLCParallel/ReflecPhase.png', bbox_inches='tight', dpi=600)
# plt.savefig('code/img/RLCParallel/ReflecPhase.svg')
plt.show()
