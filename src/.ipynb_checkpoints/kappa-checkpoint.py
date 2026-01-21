import numpy as np
from scipy.interpolate import RegularGridInterpolator
from density import CSM_density

# Low T opacity table from Ferguson et al, T < 10000 K
raw_Ferguson = np.loadtxt('optical_depth/Ferguson.txt')
logOpacity_Ferguson = raw_Ferguson[1:,1:]
logR_Ferguson = raw_Ferguson[0,1:]
logT_Ferguson = raw_Ferguson[1:,0]
logOpa_Ferguson_interp = RegularGridInterpolator((logT_Ferguson, logR_Ferguson),logOpacity_Ferguson,method='linear', bounds_error=False, fill_value=None)  # (logT,logR)

# Low T opacity table from Ferguson et al, T < 10000 K
raw_OPAL = np.loadtxt('optical_depth/OPAL.txt')
logOpacity_OPAL = raw_OPAL[1:,1:]
logR_OPAL = raw_OPAL[0,1:]
logT_OPAL = raw_OPAL[1:,0]
logOpa_OPAL_interp = RegularGridInterpolator((logT_OPAL, logR_OPAL),logOpacity_OPAL,method='linear', bounds_error=False, fill_value=None) # (logT,logR)                                         method='linear', 

def logOpacity(logT,logR):
    """
    Get the log10 value of the opacity kappa

    In
    ---
    logT : scalar
        Log10 value of temeprature
    logR : scalar
        Log10 value of R

    Out
    ---
    out : scalar
        Log10 value of opacity kappa from
    """
    if logT <= 4: # T <= 10000 K, apply Ferguson et al low T opacity table
        log_kappa = logOpa_Ferguson_interp((logT,logR))
    else: # T > 10000 K, apply OPAL table
        log_kappa = logOpa_OPAL_interp((logT,logR))
    return log_kappa

def get_logR(rho,T):
    return np.log10(rho) - 3 * np.log10(T) + 18

def kappa(rho,T):
    """
    Get the opacity kappa for a medium

    In
    ---
    rho : scalar
        Medium density, g/cm^3
    T : scalar
       Temperature, K

    Out
    ---
    out : scalar
        Opacity kappa, cm^2/g
    """
    logT = np.log10(T)
    logR = get_logR(rho,T)
    if logR < -8:
        logR = -8
    kappa = 10**logOpacity(logT,logR)
    return kappa