import numpy as np
import matplotlib.pyplot as plt
from functools import partial

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble" : r'\usepackage{siunitx}',
    "font.family": "Computer Modern Serif"
})

def dGammaNum(L, Cp, z0, ron, rhoi, piV, cc):
    dGamma = []
    for pi in piV:
        ROn = (pi/(1 + pi))*ron
        rho = rhoi*(1+pi)/(rhoi + pi)

        w0 = 1/np.sqrt(L*(Cp + cc))

        zi = (1/(1j*w0*cc) + 1/(1/(1j*w0*L) + 1j*w0*Cp + 1/ROn))
        zf = (1/(1j*w0*cc) + 1/(1/(1j*w0*L) + 1j*w0*Cp + 1/(rho*ROn)))

        dGamma.append(np.max(np.abs(
            ((z0 - zi)/(z0 + zi)) - ((z0 - zf)/(z0 + zf))
        )))

    return dGamma


z0 = 50 # Ohm
Cp = 0.5e-12 # F
ccV = np.geomspace(1e-18, 1e-3, 10001)
L = 180e-9 # H

RON = 50e3 # Ohm
rho = 2e10 # Ohm



piV = np.geomspace(1e-7, 1e15, 1001)


fig, ax = plt.subplots(1, 2)
fig.subplots_adjust(wspace=0.3, left=0.10, right=0.95, top=0.98, bottom=0.15)
dGNum = dGammaNum(L, Cp, z0, rho, RON, piV, ccV)
dGApprox = 2*np.abs((1 - np.sqrt(1 + piV))/(1 + np.sqrt(1 + piV)))

# aspect = 1

ax[0].plot(piV, dGNum, label='Numeric Contrast')
ax[0].plot(piV, dGApprox, label='Approx Contrast')
ax[0].set_xscale('log')
ax[0].legend(loc='upper left')
ax[0].set_xlabel(r'\(\pi\)')
ax[0].set_ylabel(r'\(|\Delta\Gamma|\)')
# ax[0].set_aspect(1)

ax[1].plot(piV, np.abs(dGNum-dGApprox))
ax[1].set_xscale('log')
# ax[1].set_yscale('log')
ax[1].set_xlabel(r'\(\pi\)')
ax[1].set_ylabel(r'\(|\Delta\Gamma|_\text{Aprox}-|\Delta\Gamma|_{\text{Num}}\)')
# ax[1].set_ylabel("||ΔΓ|Aprox - |ΔΓ|Num|")
# ax[1].set_aspect(1)

# ax.set_title("Contrast variation with L and Cc")
# ax.set_xlabel("L")
# ax.set_ylabel("Cc")

plt.show()
