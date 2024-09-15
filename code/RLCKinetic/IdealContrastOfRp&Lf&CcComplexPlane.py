import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from numba import njit, prange
from time import process_time, sleep
from IdealContrastDGamma import maxComplexdGammaOfLambda, maxComplexdGammaOfPi, complexdGammaOfCc

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble" : r'\usepackage{siunitx}',
    "font.family": "Computer Modern Serif"
})

plotType = "polar"
# plotType = "cartesian"

z0 = 50.0 # Ohm
cp = 0.5e-12 # Farad

ri = 50e3 # Ohm
rho = 2e6

li = 180e-9 # Henry
# lamb = 0.000000001
lamb = 0.1


npi = 101
nlambda = 101
ncc = 3501


piV = np.geomspace(1e-7, 1e7, npi)
# lambdaV = np.geomspace(lamb, 1, nlambda)
lambdaV = np.log10(np.linspace(10**lamb, 10, nlambda))
# ccV = np.geomspace(1e-18, 5e-4, ncc)
ccV = np.geomspace(1e-18, 1e-3, ncc)


#Max |dGamma(Lambda)| = dGL{case}
dGLLiF = lambda pi: maxComplexdGammaOfLambda(z0, cp, ri, li, rho, lambdaV, 0, pi, ccV, 0.0)
dGLMidF = lambda pi, wContrl: maxComplexdGammaOfLambda(z0, cp, ri, li, rho, lambdaV, 1, pi, ccV, wContrl)
dGLLfF = lambda pi: maxComplexdGammaOfLambda(z0, cp, ri, li, rho, lambdaV, 2, pi, ccV, 0.0)

#Max |dGamma(Pi)| = dGP{case}
dGPRef = maxComplexdGammaOfPi(z0, cp, ri, li, rho, 1, 0, piV, ccV, 0.0)
dGPLiF = lambda lamb: maxComplexdGammaOfPi(z0, cp, ri, li, rho, lamb, 0, piV, ccV, 0.0)
dGPMidF = lambda lamb, wContrl: maxComplexdGammaOfPi(z0, cp, ri, li, rho, lamb, 1, piV, ccV, wContrl)
dGPLfF = lambda lamb: maxComplexdGammaOfPi(z0, cp, ri, li, rho, lamb, 2, piV, ccV, 0.0)

#|dGamma(Cc)| = dGCc{case}
dGCcRefF = lambda pi: complexdGammaOfCc(z0, cp, ri, li, rho, 1, 0, pi, ccV, 0.0)
dGCcLiF = lambda pi, lamb: complexdGammaOfCc(z0, cp, ri, li, rho, lamb, 0, pi, ccV, 0.0)
dGCcMidF = lambda pi, lamb, wContrl: complexdGammaOfCc(z0, cp, ri, li, rho, lamb, 1, pi, ccV, wContrl)
dGCcLfF = lambda pi, lamb: complexdGammaOfCc(z0, cp, ri, li, rho, lamb, 2, pi, ccV, 0.0)


piI = npi//2
lambdaI = 0
lambdaTI = 0.0

if plotType == "cartesian":
    fig, ax = plt.subplots(1, 3)
elif plotType == "polar":
    fig, ax = plt.subplots(1, 3, subplot_kw={'projection': 'polar'})
else:
    fig, ax = plt.subplots(1, 3, subplot_kw={'projection': 'polar'})

# ax[1].tick_params(axis='y', which='both',  left=False)
# ax[1].set_yticklabels([])
# # ax[2].tick_params(axis='y', which='both',  left=False)
# # ax[2].set_yticklabels([])
#
# ax[2].yaxis.set_label_position('right')
# ax[2].yaxis.tick_right()


start = process_time()

dGLLi = dGLLiF(piV[piI])
dGLLf = dGLLfF(piV[piI])
dGLMid = dGLMidF(piV[piI], lambdaTI)

dGPLi = dGPLiF(lambdaV[lambdaI])
dGPLf = dGPLfF(lambdaV[lambdaI])
dGPMid = dGPMidF(lambdaV[lambdaI], lambdaTI)

print(process_time() - start)
dGCcLi = dGCcLiF(piV[piI], lambdaV[lambdaI])
dGCcLf = dGCcLfF(piV[piI], lambdaV[lambdaI])
dGCcMid = dGCcMidF(piV[piI], lambdaV[lambdaI], lambdaTI)
dGCcRef = dGCcRefF(piV[piI])

if plotType == "cartesian":
    contrLi0, = ax[0].plot(np.real(dGLLi), np.imag(dGLLi))
    contrLf0, = ax[0].plot(np.real(dGLLf), np.imag(dGLLf))
    contrMid0, = ax[0].plot(np.real(dGLMid), np.imag(dGLMid))

    contrLi1, = ax[1].plot(np.real(dGPLi), np.imag(dGPLi), zorder=1)
    contrLf1, = ax[1].plot(np.real(dGPLf), np.imag(dGPLf), zorder=2)
    contrMid1, = ax[1].plot(np.real(dGPMid), np.imag(dGPMid), zorder=3)
    contrRef1, = ax[1].plot(np.real(dGPRef), np.imag(dGPRef), zorder=0)

    contrLi2, = ax[2].plot(np.real(dGCcLi), np.imag(dGCcLi), label=r'\(\lambda_t=1\)')
    contrLf2, = ax[2].plot(np.real(dGCcLf), np.imag(dGCcLf), label=r'\(\lambda_t=\lambda\)')
    contrMid2, = ax[2].plot(np.real(dGCcMid), np.imag(dGCcMid), label=r'\(\lambda<\lambda_t<1\)')
    contrRef2, = ax[2].plot(np.real(dGCcRef), np.imag(dGCcRef), zorder=0, label=r'\(\lambda=1\)')
elif plotType == "polar":
    contrLi0, = ax[0].plot(np.angle(dGLLi), np.abs(dGLLi))
    contrLf0, = ax[0].plot(np.angle(dGLLf), np.abs(dGLLf))
    contrMid0, = ax[0].plot(np.angle(dGLMid), np.abs(dGLMid))

    contrLi1, = ax[1].plot(np.angle(dGPLi), np.abs(dGPLi), zorder=1)
    contrLf1, = ax[1].plot(np.angle(dGPLf), np.abs(dGPLf), zorder=2)
    contrMid1, = ax[1].plot(np.angle(dGPMid), np.abs(dGPMid), zorder=3)
    contrRef1, = ax[1].plot(np.angle(dGPRef), np.abs(dGPRef), zorder=0)

    contrLi2, = ax[2].plot(np.angle(dGCcLi), np.abs(dGCcLi), label=r'\(\lambda_t=1\)')
    contrLf2, = ax[2].plot(np.angle(dGCcLf), np.abs(dGCcLf), label=r'\(\lambda_t=\lambda\)')
    contrMid2, = ax[2].plot(np.angle(dGCcMid), np.abs(dGCcMid), label=r'\(\lambda<\lambda_t<1\)')
    contrRef2, = ax[2].plot(np.angle(dGCcRef), np.abs(dGCcRef), zorder=0, label=r'\(\lambda=1\)')
else:
    contrLi0, = ax[0].plot(np.angle(dGLLi), np.abs(dGLLi))
    contrLf0, = ax[0].plot(np.angle(dGLLf), np.abs(dGLLf))
    contrMid0, = ax[0].plot(np.angle(dGLMid), np.abs(dGLMid))

    contrLi1, = ax[1].plot(np.angle(dGPLi), np.abs(dGPLi), zorder=1)
    contrLf1, = ax[1].plot(np.angle(dGPLf), np.abs(dGPLf), zorder=2)
    contrMid1, = ax[1].plot(np.angle(dGPMid), np.abs(dGPMid), zorder=3)
    contrRef1, = ax[1].plot(np.angle(dGPRef), np.abs(dGPRef), zorder=0)

    contrLi2, = ax[2].plot(np.angle(dGCcLi), np.abs(dGCcLi), label=r'\(\lambda_t=1\)')
    contrLf2, = ax[2].plot(np.angle(dGCcLf), np.abs(dGCcLf), label=r'\(\lambda_t=\lambda\)')
    contrMid2, = ax[2].plot(np.angle(dGCcMid), np.abs(dGCcMid), label=r'\(\lambda<\lambda_t<1\)')
    contrRef2, = ax[2].plot(np.angle(dGCcRef), np.abs(dGCcRef), zorder=0, label=r'\(\lambda=1\)')



# ax[0].grid(axis='y')
# ax[1].grid(axis='y')
# ax[2].grid(axis='y')
if plotType == 'cartesian':
    for axe in ax:
        axe.set_xlim(-2, 2)
        axe.set_ylim(-2, 2)
        axe.set_box_aspect(1)
elif plotType == 'polar':
    for axe in ax:
        axe.set_rmax(2)
        axe.set_rticks([0.5, 1, 1.5])
        axe.set_rlabel_position(135)
else:
    for axe in ax:
        axe.set_rmax(2)
        axe.set_rticks([0.5, 1, 1.5])
        axe.set_rlabel_position(135)
ax[0].set_title(r'\(\Delta\Gamma_{\text{Opt}}(\lambda)\)')
ax[1].set_title(r'\(\Delta\Gamma_{\text{Opt}}(\pi)\)')
ax[2].set_title(r'\(\Delta\Gamma(C_c)\)')
# ax[0].set_ylabel("Max |ΔΓ|")
# ax[2].set_ylabel("|ΔΓ|")
# ax[0].set_xscale("log")
# ax[1].set_xscale("log")
# ax[2].set_xscale("log")
# ax[0].set_xlabel("λ")
# ax[1].set_xlabel("π")
# ax[2].set_xlabel("Cc")
ax[2].legend(loc='upper right')
#
fig.subplots_adjust(wspace=0.3)


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

    if plotType == 'cartesian':
        contrLi0.set_xdata(np.real(dGLLi))
        contrLi0.set_ydata(np.imag(dGLLi))
        contrLf0.set_xdata(np.real(dGLLf))
        contrLf0.set_ydata(np.imag(dGLLf))
        contrMid0.set_xdata(np.real(dGLMid))
        contrMid0.set_ydata(np.imag(dGLMid))

        contrLi1.set_xdata(np.real(dGPLi))
        contrLi1.set_ydata(np.imag(dGPLi))
        contrLf1.set_xdata(np.real(dGPLf))
        contrLf1.set_ydata(np.imag(dGPLf))
        contrMid1.set_xdata(np.real(dGPMid))
        contrMid1.set_ydata(np.imag(dGPMid))

        contrLi2.set_xdata(np.real(dGCcLi))
        contrLi2.set_ydata(np.imag(dGCcLi))
        contrLf2.set_xdata(np.real(dGCcLf))
        contrLf2.set_ydata(np.imag(dGCcLf))
        contrMid2.set_xdata(np.real(dGCcMid))
        contrMid2.set_ydata(np.imag(dGCcMid))
        contrRef2.set_xdata(np.real(dGCcRef))
        contrRef2.set_ydata(np.imag(dGCcRef))
    elif plotType == 'polar':
        contrLi0.set_xdata(np.angle(dGLLi))
        contrLi0.set_ydata(np.abs(dGLLi))
        contrLf0.set_xdata(np.angle(dGLLf))
        contrLf0.set_ydata(np.abs(dGLLf))
        contrMid0.set_xdata(np.angle(dGLMid))
        contrMid0.set_ydata(np.abs(dGLMid))

        contrLi1.set_xdata(np.angle(dGPLi))
        contrLi1.set_ydata(np.abs(dGPLi))
        contrLf1.set_xdata(np.angle(dGPLf))
        contrLf1.set_ydata(np.abs(dGPLf))
        contrMid1.set_xdata(np.angle(dGPMid))
        contrMid1.set_ydata(np.abs(dGPMid))

        contrLi2.set_xdata(np.angle(dGCcLi))
        contrLi2.set_ydata(np.abs(dGCcLi))
        contrLf2.set_xdata(np.angle(dGCcLf))
        contrLf2.set_ydata(np.abs(dGCcLf))
        contrMid2.set_xdata(np.angle(dGCcMid))
        contrMid2.set_ydata(np.abs(dGCcMid))
        contrRef2.set_xdata(np.angle(dGCcRef))
        contrRef2.set_ydata(np.abs(dGCcRef))
    else:
        contrLi0.set_xdata(np.angle(dGLLi))
        contrLi0.set_ydata(np.abs(dGLLi))
        contrLf0.set_xdata(np.angle(dGLLf))
        contrLf0.set_ydata(np.abs(dGLLf))
        contrMid0.set_xdata(np.angle(dGLMid))
        contrMid0.set_ydata(np.abs(dGLMid))

        contrLi1.set_xdata(np.angle(dGPLi))
        contrLi1.set_ydata(np.abs(dGPLi))
        contrLf1.set_xdata(np.angle(dGPLf))
        contrLf1.set_ydata(np.abs(dGPLf))
        contrMid1.set_xdata(np.angle(dGPMid))
        contrMid1.set_ydata(np.abs(dGPMid))

        contrLi2.set_xdata(np.angle(dGCcLi))
        contrLi2.set_ydata(np.abs(dGCcLi))
        contrLf2.set_xdata(np.angle(dGCcLf))
        contrLf2.set_ydata(np.abs(dGCcLf))
        contrMid2.set_xdata(np.angle(dGCcMid))
        contrMid2.set_ydata(np.abs(dGCcMid))
        contrRef2.set_xdata(np.angle(dGCcRef))
        contrRef2.set_ydata(np.abs(dGCcRef))


    fig.canvas.draw_idle()
    return


def updateW(val):
    piI = int(pi_slider.val)
    lambdaI = int(lambda_slider.val)
    lambdaTI = lambdaT_slider.val

    lambdaT_slider.valtext.set_text('{:.2e}'.format(np.sqrt(lambdaV[lambdaI])**(1-lambdaTI)))

    dGLMid = dGLMidF(piV[piI], lambdaTI)
    dGPMid = dGPMidF(lambdaV[lambdaI], lambdaTI)
    dGCcMid = dGCcMidF(piV[piI], lambdaV[lambdaI], lambdaTI)

    if plotType == 'cartesian':
        contrMid0.set_xdata(np.real(dGLMid))
        contrMid0.set_ydata(np.abs(dGLMid))

        contrMid1.set_xdata(np.real(dGPMid))
        contrMid1.set_ydata(np.abs(dGPMid))

        contrMid2.set_xdata(np.real(dGCcMid))
        contrMid2.set_ydata(np.abs(dGCcMid))
    elif plotType == 'polar':
        contrMid0.set_xdata(np.angle(dGLMid))
        contrMid0.set_ydata(np.abs(dGLMid))

        contrMid1.set_xdata(np.angle(dGPMid))
        contrMid1.set_ydata(np.abs(dGPMid))

        contrMid2.set_xdata(np.angle(dGCcMid))
        contrMid2.set_ydata(np.abs(dGCcMid))
    else:
        contrMid0.set_xdata(np.angle(dGLMid))
        contrMid0.set_ydata(np.abs(dGLMid))

        contrMid1.set_xdata(np.angle(dGPMid))
        contrMid1.set_ydata(np.abs(dGPMid))

        contrMid2.set_xdata(np.angle(dGCcMid))
        contrMid2.set_ydata(np.abs(dGCcMid))



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
# plt.savefig('code/img/KineticContr/DoubleContrastPlot.png'.format(lamb), bbox_inches='tight', dpi=600)

plt.show()
