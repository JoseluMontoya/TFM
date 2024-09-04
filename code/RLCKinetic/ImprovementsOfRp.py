import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from numba import njit, prange
from time import process_time, sleep
from IdealContrastDGamma import maxdGammaOfLambda, maxdGammaOfPi, dGammaOfCc

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble" : r'\usepackage{siunitx}',
    "font.family": "Computer Modern Serif"
})

z0 = 50.0 # Ohm
cp = 0.5e-12 # Farad

ri = 50e3 # Ohm
# ri = 4e3 # Ohm
rho = 2e6

li = 180e-9 # Henry
# lamb = 0.000000001
lamb = 0.1


npi = 10001
nlambda = 101
ncc = 2001


piV = np.geomspace(1e-7, 1e7, npi)
# lambdaV = np.geomspace(lamb, 1, nlambda)
lambdaV = np.log10(np.linspace(10**lamb, 10, nlambda))
# ccV = np.geomspace(1e-18, 5e-4, ncc)
ccV = np.geomspace(1e-18, 1e-3, ncc)
# ccV = np.geomspace(1e-18, 1e3, ncc)


#Max |dGamma(Pi)| = dGP{case}
dGPRef = maxdGammaOfPi(z0, cp, ri, li, rho, 1, 0, piV, ccV, 0.0)
dGPLiF = lambda lamb: maxdGammaOfPi(z0, cp, ri, li, rho, lamb, 0, piV, ccV, 0.0)
dGPMidF = lambda lamb, wContrl: maxdGammaOfPi(z0, cp, ri, li, rho, lamb, 1, piV, ccV, wContrl)
dGPLfF = lambda lamb: maxdGammaOfPi(z0, cp, ri, li, rho, lamb, 2, piV, ccV, 0.0)

piI = npi//2
lambdaI = 0
lambdaTI = 0.0


fig, ax = plt.subplots()
# fig, ax = plt.subplots(1, 3)
# ax[1].set_yticklabels([])
# ax[2].tick_params(axis='y', which='both',  left=False)
# ax[2].set_yticklabels([])

start = process_time()


dGPLi = dGPLiF(lambdaV[lambdaI])
dGPLf = dGPLfF(lambdaV[lambdaI])
dGPMid = dGPMidF(lambdaV[lambdaI], lambdaTI)


contrLi1, = ax.plot(piV, dGPLiF(lambdaV[lambdaI])-dGPRef, zorder=1, label=r'\(\lambda_t=1\)')
contrLf1, = ax.plot(piV, dGPLfF(lambdaV[lambdaI])-dGPRef, zorder=2, label=r'\(\lambda_t=\lambda\)')
contrMid1, = ax.plot(piV, dGPMidF(lambdaV[lambdaI], lambdaTI)-dGPRef, zorder=3, label=r'\(\lambda<\lambda_t<1\)')
# contrRef1, = ax[1].plot(piV, 2*np.abs((1-np.sqrt(1+piV))/(1+np.sqrt(1+piV))), zorder=0)

ax.grid(axis='y')
# ax.set_ylim(0, 2)
ax.set_ylabel(r'\(\Delta|\Delta\Gamma|\)')
ax.set_xscale('log')
ax.set_xlabel(r'\(\pi\)')
ax.legend(loc='upper right')

fig.subplots_adjust(right=0.979, left=0.049, bottom=0.193, top=0.956, wspace=0)

axLambda = fig.add_axes((0.17, 0.045, 0.65, 0.03))
lambda_slider = Slider(
    ax=axLambda,
    # label=r'\(\rho\)',
    label=r'\(\lambda\)',
    valmin=0,
    valmax=nlambda-1,
    valinit=lambdaI,
    valstep=1,
    )
lambda_slider.valtext.set_text('{:.2e}'.format(lambdaV[lambdaI]))


axLambdaT= fig.add_axes((0.17, 0.015, 0.65, 0.03))
lambdaT_slider = Slider(
    ax=axLambdaT,
    # label=r'\(\rho\)',
    label=r'\(\lambda_t\)',
    valmin=-1,
    valmax=1,
    valinit=lambdaTI,
    valstep=0.001,
    )
lambdaT_slider.valtext.set_text('{:.2e}'.format(np.sqrt(lambdaV[lambdaI])**(1 - lambdaTI)))



def update(val):
    lambdaTI = lambdaT_slider.val
    lambdaI = int(lambda_slider.val)

    lambda_slider.valtext.set_text('{:.2e}'.format(lambdaV[lambdaI]))
    lambdaT_slider.valtext.set_text('{:.2e}'.format(np.sqrt(lambdaV[lambdaI])**(1-lambdaTI)))


    contrLi1.set_ydata(dGPLiF(lambdaV[lambdaI])-dGPRef)
    contrLf1.set_ydata(dGPLfF(lambdaV[lambdaI])-dGPRef)
    contrMid1.set_ydata(dGPMidF(lambdaV[lambdaI], lambdaTI)-dGPRef)

    fig.canvas.draw_idle()
    return


def updateW(val):
    lambdaI = int(lambda_slider.val)
    lambdaTI = lambdaT_slider.val

    lambdaT_slider.valtext.set_text('{:.2e}'.format(np.sqrt(lambdaV[lambdaI])**(1-lambdaTI)))

    contrMid1.set_ydata(dGPMidF(lambdaV[lambdaI], lambdaTI)-dGPRef)

    fig.canvas.draw_idle()
    return

lambda_slider.on_changed(update)
lambdaT_slider.on_changed(updateW)
# #

resetax = fig.add_axes((0.02, 0.04, 0.1, 0.04))
button = Button(resetax, 'Reset', hovercolor='0.975')
def reset(event):
    lambda_slider.reset()
    lambdaT_slider.reset()
    return
button.on_clicked(reset)
# # plt.savefig('code/img/KineticContr/DoubleContrastPlot.png'.format(lamb), bbox_inches='tight', dpi=600)
plt.show()
