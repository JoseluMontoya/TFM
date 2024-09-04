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


npi = 101
nlambda = 101
ncc = 2001


piV = np.geomspace(1e-7, 1e7, npi)
# lambdaV = np.geomspace(lamb, 1, nlambda)
lambdaV = np.log10(np.linspace(10**lamb, 10, nlambda))
# ccV = np.geomspace(1e-18, 5e-4, ncc)
ccV = np.geomspace(1e-18, 1e-3, ncc)
# ccV = np.geomspace(1e-18, 1e3, ncc)


#Max |dGamma(Lambda)| = dGL{case}
dGLLiF = lambda pi: maxdGammaOfLambda(z0, cp, ri, li, rho, lambdaV, 0, pi, ccV, 0.0)
dGLMidF = lambda pi, wContrl: maxdGammaOfLambda(z0, cp, ri, li, rho, lambdaV, 1, pi, ccV, wContrl)
dGLLfF = lambda pi: maxdGammaOfLambda(z0, cp, ri, li, rho, lambdaV, 2, pi, ccV, 0.0)

#Max |dGamma(Pi)| = dGP{case}
dGPRef = maxdGammaOfPi(z0, cp, ri, li, rho, 1, 0, piV, ccV, 0.0)
dGPLiF = lambda lamb: maxdGammaOfPi(z0, cp, ri, li, rho, lamb, 0, piV, ccV, 0.0)
dGPMidF = lambda lamb, wContrl: maxdGammaOfPi(z0, cp, ri, li, rho, lamb, 1, piV, ccV, wContrl)
dGPLfF = lambda lamb: maxdGammaOfPi(z0, cp, ri, li, rho, lamb, 2, piV, ccV, 0.0)

#|dGamma(Cc)| = dGCc{case}
dGCcRefF = lambda pi: dGammaOfCc(z0, cp, ri, li, rho, 1, 0, pi, ccV, 0.0)
dGCcLiF = lambda pi, lamb: dGammaOfCc(z0, cp, ri, li, rho, lamb, 0, pi, ccV, 0.0)
dGCcMidF = lambda pi, lamb, wContrl: dGammaOfCc(z0, cp, ri, li, rho, lamb, 1, pi, ccV, wContrl)
dGCcLfF = lambda pi, lamb: dGammaOfCc(z0, cp, ri, li, rho, lamb, 2, pi, ccV, 0.0)


piI = npi//2
lambdaI = 0
lambdaTI = 0.0


fig, ax = plt.subplots(1, 3, sharey= True)
# fig, ax = plt.subplots(1, 3)
ax[1].tick_params(axis='y', which='both',  left=False)
# ax[1].set_yticklabels([])
# ax[2].tick_params(axis='y', which='both',  left=False)
# ax[2].set_yticklabels([])

ax[2].yaxis.set_label_position('right')
ax[2].yaxis.tick_right()

start = process_time()

dGLLi = dGLLiF(piV[piI])
dGLLf = dGLLfF(piV[piI])
dGLMid = dGLMidF(piV[piI], lambdaTI)

dGPLi = dGPLiF(lambdaV[lambdaI])
dGPLf = dGPLfF(lambdaV[lambdaI])
dGPMid = dGPMidF(lambdaV[lambdaI], lambdaTI)

dGCcLi = dGCcLiF(piV[piI], lambdaV[lambdaI])
dGCcLf = dGCcLfF(piV[piI], lambdaV[lambdaI])
dGCcMid = dGCcMidF(piV[piI], lambdaV[lambdaI], lambdaTI)
dGCcRef = dGCcRefF(piV[piI])
print(process_time() - start)


lambdaline = ax[0].axvline(lambdaV[lambdaI], c='k', linestyle='dotted', zorder=-1)
contrLi0, = ax[0].plot(lambdaV, dGLLiF(piV[piI]))
contrLf0, = ax[0].plot(lambdaV, dGLLfF(piV[piI]))
contrMid0, = ax[0].plot(lambdaV, dGLMidF(piV[piI], lambdaTI))

piline = ax[1].axvline(piV[piI], c='k', linestyle='dotted', zorder=-1)
contrLi1, = ax[1].plot(piV, dGPLiF(lambdaV[lambdaI]), zorder=1)
contrLf1, = ax[1].plot(piV, dGPLfF(lambdaV[lambdaI]), zorder=2)
contrMid1, = ax[1].plot(piV, dGPMidF(lambdaV[lambdaI], lambdaTI), zorder=3)
# contrRef1, = ax[1].plot(piV, dGPRef, zorder=0)
contrRef1, = ax[1].plot(piV, 2*np.abs((1-np.sqrt(1+piV))/(1+np.sqrt(1+piV))), zorder=0)

contrLi2, = ax[2].plot(ccV*1e12, dGCcLiF(piV[piI], lambdaV[lambdaI]), label=r'\(\lambda_t=1\)')
contrLf2, = ax[2].plot(ccV*1e12, dGCcLfF(piV[piI], lambdaV[lambdaI]), label=r'\(\lambda_t=\lambda\)')
contrMid2, = ax[2].plot(ccV*1e12, dGCcMidF(piV[piI], lambdaV[lambdaI], lambdaTI), label=r'\(\lambda<\lambda_t<1\)')
contrRef2, = ax[2].plot(ccV*1e12, dGCcRefF(piV[piI]), zorder=0, label=r'\(\lambda=1\)')

maxCcLi2 = ax[2].axvline(1e12*ccV[np.argmax(dGCcLiF(piV[piI], lambdaV[lambdaI]))], ymax=np.max(dGCcLiF(piV[piI], lambdaV[lambdaI]))/2, c='C0', zorder=1, linestyle='dotted')
maxCcLf2 = ax[2].axvline(1e12*ccV[np.argmax(dGCcLfF(piV[piI], lambdaV[lambdaI]))], ymax=np.max(dGCcLfF(piV[piI], lambdaV[lambdaI]))/2, c='C1', zorder=2, linestyle='dotted')
maxCcMid2 = ax[2].axvline(1e12*ccV[np.argmax(dGCcMidF(piV[piI], lambdaV[lambdaI], lambdaTI))], ymax=np.max(dGCcMidF(piV[piI], lambdaV[lambdaI], lambdaTI))/2, c='C2', zorder=3, linestyle='dotted')
maxCcRef2 = ax[2].axvline(1e12*ccV[np.argmax(dGCcRefF(piV[piI]))], ymax=np.max(dGCcRefF(piV[piI]))/2, c='C3', zorder=0, linestyle='dotted')


ax[0].grid(axis='y')
ax[1].grid(axis='y')
ax[2].grid(axis='y')
ax[0].set_title(r'\(|\Delta\Gamma|(\lambda)\)')
ax[1].set_title(r'\(|\Delta\Gamma|(\pi)\)')
ax[2].set_title(r'\(|\Delta\Gamma|(C_c)\)')
ax[0].set_ylim(0, 2)
ax[1].set_ylim(0, 2)
ax[2].set_ylim(0, 2)
ax[0].set_ylabel(r'Max \(|\Delta\Gamma|\)')
ax[2].set_ylabel(r'\(|\Delta\Gamma|\)')
# ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[2].set_xscale('log')
ax[0].set_xlabel(r'\(\lambda\)')
ax[1].set_xlabel(r'\(\pi\)')
ax[2].set_xlabel(r'\(C_c(\unit{\pico\farad})\)')
ax[2].legend(loc='upper right')

fig.subplots_adjust(right=0.979, left=0.049, bottom=0.193, top=0.956, wspace=0)

axPi = fig.add_axes((0.17, 0.075, 0.65, 0.03))
pi_slider = Slider(
    ax=axPi,
    # label=r'\(\rho\)',
    label=r'\(\pi\)',
    valmin=0,
    valmax=npi-1,
    valinit=piI,
    valstep=1,
    )
pi_slider.valtext.set_text('{:.2e}'.format(piV[piI]))

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
    piI = int(pi_slider.val)
    lambdaI = int(lambda_slider.val)

    pi_slider.valtext.set_text('{:.2e}'.format(piV[piI]))
    lambda_slider.valtext.set_text('{:.2e}'.format(lambdaV[lambdaI]))
    lambdaT_slider.valtext.set_text('{:.2e}'.format(np.sqrt(lambdaV[lambdaI])**(1-lambdaTI)))

    contrLi0.set_ydata(dGLLiF(piV[piI]))
    contrLf0.set_ydata(dGLLfF(piV[piI]))
    contrMid0.set_ydata(dGLMidF(piV[piI], lambdaTI))
    piline.set_xdata(piV[piI]*np.ones(2))

    contrLi1.set_ydata(dGPLiF(lambdaV[lambdaI]))
    contrLf1.set_ydata(dGPLfF(lambdaV[lambdaI]))
    contrMid1.set_ydata(dGPMidF(lambdaV[lambdaI], lambdaTI))
    lambdaline.set_xdata(lambdaV[lambdaI]*np.ones(2))

    contrLi2.set_ydata(dGCcLiF(piV[piI], lambdaV[lambdaI]))
    contrLf2.set_ydata(dGCcLfF(piV[piI], lambdaV[lambdaI]))
    contrMid2.set_ydata(dGCcMidF(piV[piI], lambdaV[lambdaI], lambdaTI))
    contrRef2.set_ydata(dGCcRefF(piV[piI]))

    maxCcLi2.set_xdata([1e12*ccV[np.argmax(dGCcLiF(piV[piI], lambdaV[lambdaI]))]])
    maxCcLf2.set_xdata([1e12*ccV[np.argmax(dGCcLfF(piV[piI], lambdaV[lambdaI]))]])
    maxCcMid2.set_xdata([1e12*ccV[np.argmax(dGCcMidF(piV[piI], lambdaV[lambdaI], lambdaTI))]])
    maxCcRef2.set_xdata([1e12*ccV[np.argmax(dGCcRefF(piV[piI]))]])
    maxCcLi2.set_ydata([0, np.max(dGCcLiF(piV[piI], lambdaV[lambdaI]))/2])
    maxCcLf2.set_ydata([0, np.max(dGCcLfF(piV[piI], lambdaV[lambdaI]))/2])
    maxCcMid2.set_ydata([0, np.max(dGCcMidF(piV[piI], lambdaV[lambdaI], lambdaTI))/2])
    maxCcRef2.set_ydata([0, np.max(dGCcRefF(piV[piI]))/2])


    fig.canvas.draw_idle()
    return


def updateW(val):
    piI = int(pi_slider.val)
    lambdaI = int(lambda_slider.val)
    lambdaTI = lambdaT_slider.val

    lambdaT_slider.valtext.set_text('{:.2e}'.format(np.sqrt(lambdaV[lambdaI])**(1-lambdaTI)))

    contrMid0.set_ydata(dGLMidF(piV[piI], lambdaTI))
    contrMid1.set_ydata(dGPMidF(lambdaV[lambdaI], lambdaTI))
    contrMid2.set_ydata(dGCcMidF(piV[piI], lambdaV[lambdaI], lambdaTI))

    maxCcMid2.set_xdata([1e12*ccV[np.argmax(dGCcMidF(piV[piI], lambdaV[lambdaI], lambdaTI))]])
    maxCcMid2.set_ydata([0, np.max(dGCcMidF(piV[piI], lambdaV[lambdaI], lambdaTI))/2])


    fig.canvas.draw_idle()
    return

lambda_slider.on_changed(update)
pi_slider.on_changed(update)
lambdaT_slider.on_changed(updateW)
# #

resetax = fig.add_axes((0.02, 0.04, 0.1, 0.04))
button = Button(resetax, 'Reset', hovercolor='0.975')
def reset(event):
    pi_slider.reset()
    lambda_slider.reset()
    lambdaT_slider.reset()
    return
button.on_clicked(reset)
# # plt.savefig('code/img/KineticContr/DoubleContrastPlot.png'.format(lamb), bbox_inches='tight', dpi=600)
plt.show()
