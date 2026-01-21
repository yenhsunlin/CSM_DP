import numpy as np
from scipy.interpolate import RegularGridInterpolator,interp1d
from scipy.integrate import quad
from scipy.optimize import brentq,root_scalar
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

### ---------- Define constants ---------- ###
mH = 1.37355e-24 # hydrogen mass, g
mHe = 6.64647e-24 # helium mass, g
c = 3e10 # light speed, cm/s
a = 7.56573e-15 # radiation density constant
ME = 9.109e-28       # g
KB = 1.3806e-16      # erg/K
HBAR = 1.0545e-27    # erg*s
EV_TO_ERG = 1.60218e-12


### ---------- Data Loading ---------- ###

# --- CSM density --- #
# In the following we use realistic CSM density from Zimmerman+ Nature 627, 762 (2024)
rho_SN2020tlf_data = np.log10(np.loadtxt(BASE_DIR / 'density/SN2020tlf.txt').T)
rho_SN2023ixf_data = np.log10(np.loadtxt(BASE_DIR / 'density/SN2023ixf.txt').T)
rho_SN2023ixf_sharp_data = np.log10(np.loadtxt(BASE_DIR / 'density/SN2023ixf_sharp.txt').T)
rho_SN2024ggi_data = np.log10(np.loadtxt(BASE_DIR / 'density/SN2024ggi.txt').T)
# Interp in log10 scale
rho_2020tlf_interp = interp1d(rho_SN2020tlf_data[0],rho_SN2020tlf_data[1],kind='linear',bounds_error=False,fill_value=np.nan)
rho_2023ixf_interp = interp1d(rho_SN2023ixf_data[0],rho_SN2023ixf_data[1],kind='linear',bounds_error=False,fill_value=np.nan)
rho_2023ixf_s_interp = interp1d(rho_SN2023ixf_sharp_data[0],rho_SN2023ixf_sharp_data[1],kind='linear',bounds_error=False,fill_value=np.nan)
rho_2024ggi_interp = interp1d(rho_SN2024ggi_data[0],rho_SN2024ggi_data[1],kind='linear',bounds_error=False,fill_value=np.nan)

# Useful dictionaries
r_mins = {'SN 2020tlf':10**rho_SN2020tlf_data[0,0],'SN 2023ixf':10**rho_SN2023ixf_data[0,0],'SN 2023ixf s':10**rho_SN2023ixf_sharp_data[0,0],'SN 2024ggi':10**rho_SN2024ggi_data[0,0]} 
r_maxs = {'SN 2020tlf':10**rho_SN2020tlf_data[0,-1],'SN 2023ixf':10**rho_SN2023ixf_data[0,-1],'SN 2023ixf s':10**rho_SN2023ixf_sharp_data[0,-1],'SN 2024ggi':10**rho_SN2024ggi_data[0,-1]} 
SN_names = ['SN 2020tlf','SN 2023ixf','SN 2024ggi']


# --- Opacity --- #
# Low T opacity table from Ferguson et al, T < 10000 K
raw_Ferguson = np.loadtxt(BASE_DIR / 'opacity/Ferguson.txt')
logOpacity_Ferguson = raw_Ferguson[1:,1:]
logR_Ferguson = raw_Ferguson[0,1:]
logT_Ferguson = raw_Ferguson[1:,0]
logOpa_Ferguson_interp = RegularGridInterpolator((logT_Ferguson, logR_Ferguson),logOpacity_Ferguson,method='linear', bounds_error=False, fill_value=None)  # (logT,logR)

# Low T opacity table from Ferguson et al, T < 10000 K
raw_OPAL = np.loadtxt(BASE_DIR / 'opacity/OPAL.txt')
logOpacity_OPAL = raw_OPAL[1:,1:]
logR_OPAL = raw_OPAL[0,1:]
logT_OPAL = raw_OPAL[1:,0]
logOpa_OPAL_interp = RegularGridInterpolator((logT_OPAL, logR_OPAL),logOpacity_OPAL,method='linear', bounds_error=False, fill_value=None) # (logT,logR)                                         method='linear', 

# --- Electron CSDA --- #
# load CSDA data
CSDA_electron = np.loadtxt(BASE_DIR / 'CSDA/electron.txt',skiprows=6).T
# interpolate, in log-scale
CSDA_e_H = interp1d(np.log10(CSDA_electron[0]),np.log10(CSDA_electron[-2])) # hydrogen
CSDA_e_He = interp1d(np.log10(CSDA_electron[0]),np.log10(CSDA_electron[-1])) # helium

# # --- Average stopping time --- #
# avg_stop_time_LS220 = np.loadtxt(BASE_DIR / 'average_stopping_time/LS220_avg_stop_time.txt')
# logAvg_stop_time = avg_stop_time_LS220[1:,1:]
# logR_avg_stop_time = avg_stop_time_LS220[0,1:]
# logmAp_avg_stop_time = avg_stop_time_LS220[1:,0]

# # interpolation
# avg_stop_time_LS220_interp = RegularGridInterpolator((logmAp_avg_stop_time, logR_avg_stop_time),logAvg_stop_time,method='linear', bounds_error=False, fill_value=None)

# --- Dark Photon Luminosity --- #
# * -- With EL distribution -- * #
# Load data and restructure it into (91,41,32)
DP_data_dQ_drdE_11_LS220 = np.loadtxt(BASE_DIR / 'dp_luminosity/time_integrated/LS220_dQ_drdE_eps_1e-11.txt')[:, 3].reshape(91,41,32)
DP_data_dQ_drdE_13_LS220 = np.loadtxt(BASE_DIR / 'dp_luminosity/time_integrated/LS220_dQ_drdE_eps_1e-13.txt')[:, 3].reshape(91,41,32)

mAp_axis = np.linspace(0,2.7,91)
r_axis = np.linspace(12,20,41)
EL_axis = np.linspace(0,2.7,32)

# eps = 1e-11 and 1e-13
DP_data_dQ_drdE_11_LS220_interp = RegularGridInterpolator((mAp_axis, r_axis, EL_axis),DP_data_dQ_drdE_11_LS220,method='linear',bounds_error=True)
DP_data_dQ_drdE_13_LS220_interp = RegularGridInterpolator((mAp_axis, r_axis, EL_axis),DP_data_dQ_drdE_13_LS220,method='linear',bounds_error=True)

# * -- With EL distribution integrated -- * #
# Load data and restructure it into (91,41,32)
DP_data_dQ_dr_11_LS220 = np.loadtxt(BASE_DIR / 'dp_luminosity/time_integrated/LS220_dQ_dr_eps_1e-11.txt')[:, 2].reshape(91,41)
DP_data_dQ_dr_13_LS220 = np.loadtxt(BASE_DIR / 'dp_luminosity/time_integrated/LS220_dQ_dr_eps_1e-13.txt')[:, 2].reshape(91,41)

# eps = 1e-11 and 1e-13
DP_data_dQ_dr_11_LS220_interp = RegularGridInterpolator((mAp_axis, r_axis),DP_data_dQ_dr_11_LS220,method='linear',bounds_error=True)
DP_data_dQ_dr_13_LS220_interp = RegularGridInterpolator((mAp_axis, r_axis),DP_data_dQ_dr_13_LS220,method='linear',bounds_error=True)


### ---------- Useful Functions ---------- ###

def CSM_density(r,SN_name='SN 2023ixf'):
    """
    CSM density for different SN
    
    Parameters
    ----------
    r : scalar
        CSM layer, cm
    SN_name : str
        Name of the SN, only three are documented: SN 2020tlf, SN 2023ixf and SN 2024ggi.

    Returns
    -------
    out : scalr
        CSM density at layer r, g/cm^3
    """
    r = np.log10(r)
    if SN_name == 'SN 2020tlf':
        return 10**rho_2020tlf_interp(r)
    elif SN_name == 'SN 2023ixf':
        return 10**rho_2023ixf_interp(r)
    elif SN_name == 'SN 2023ixf s':
        return 10**rho_2023ixf_s_interp(r)
    elif SN_name == 'SN 2024ggi':
        return 10**rho_2024ggi_interp(r)
    else:
        raise ValueError('\'SN_name\' must be a \'SN 2020tlf\', \'SN 2023ixf\' or \'SN 2024ggi\'.')

def get_logR(rho, T):
    return np.log10(rho) - 3 * np.log10(T) + 18

def logOpacity(logT, logR):
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
    # Convert inputs to array-like
    logT = np.atleast_1d(logT)
    logR = np.atleast_1d(logR)
    
    log_kappa = np.zeros_like(logT)
    
    # T > 1e4 K using OPAL, otherwise Ferguson table
    logT = np.atleast_1d(logT)
    logR = np.atleast_1d(logR)
    
    # Initialize everything as nan
    log_kappa = np.full_like(logT, np.nan, dtype=float)
    
    # Define the absolute valid range
    valid_t_mask = (logT >= np.log10(600)) & (logT <= 8.7)
    
    # Create sub-masks that ONLY apply to valid points
    # This prevents the interpolator from seeing points it can't handle
    low_t_mask = valid_t_mask & (logT <= 4)
    high_t_mask = valid_t_mask & (logT > 4)
    
    # Fill in values for Ferguson table
    if np.any(low_t_mask):
        pts_low = np.stack((logT[low_t_mask], logR[low_t_mask]), axis=-1)
        log_kappa[low_t_mask] = logOpa_Ferguson_interp(pts_low)
        
    # Fill in values for OPAL table
    if np.any(high_t_mask):
        pts_high = np.stack((logT[high_t_mask], logR[high_t_mask]), axis=-1)
        log_kappa[high_t_mask] = logOpa_OPAL_interp(pts_high)
    
    return log_kappa if log_kappa.size > 1 else log_kappa[0]

def kappa(rho, T):
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
    logR = get_logR(rho, T)
    
    # When logR < -8, truncate it at -8
    logR_clamped = np.maximum(logR, -8)
    
    # get log value of kappa
    log_k = logOpacity(logT, logR_clamped)
    
    return 10**log_k

# def logOpacity(logT,logR):
#     """
#     Get the log10 value of the opacity kappa

#     In
#     ---
#     logT : scalar
#         Log10 value of temeprature
#     logR : scalar
#         Log10 value of R

#     Out
#     ---
#     out : scalar
#         Log10 value of opacity kappa from
#     """
#     if logT <= 4: # T <= 10000 K, apply Ferguson et al low T opacity table
#         log_kappa = logOpa_Ferguson_interp((logT,logR))
#     else: # T > 10000 K, apply OPAL table
#         log_kappa = logOpa_OPAL_interp((logT,logR))
#     return log_kappa

# def get_logR(rho,T):
#     return np.log10(rho) - 3 * np.log10(T) + 18

# def kappa(rho,T):
#     """
#     Get the opacity kappa for a medium

#     In
#     ---
#     rho : scalar
#         Medium density, g/cm^3
#     T : scalar
#        Temperature, K

#     Out
#     ---
#     out : scalar
#         Opacity kappa, cm^2/g
#     """
#     logT = np.log10(T)
#     logR = get_logR(rho,T)
#     if logR < -8:
#         logR = -8
#     kappa = 10**logOpacity(logT,logR)
#     return kappa

def tau(r,T,R_max=3e14,SN_name='SN 2023ixf'):
    """
    Optical depth for different SN profile

    In
    ---
    r : scalar
        The radius of CSM, cm
    T : scalar
        The CSM temperature, K
    R : scalar
        The maximum radius of CSM, cm
    SN_name : str
        The name of the SN profile

    Out
    ---
    out : scalar
        The optical depth
    """
    def _rho_kappa(r):
        rho = CSM_density(r,SN_name)
        return kappa(rho,T) * rho
    return quad(_rho_kappa,r,R_max)[0]

def dQ_dr_LS220(eps,mAp,r):
    mAp = np.log10(mAp)
    try:
        # try reference eps = 1e-13 first
        scaling = (eps / 1e-13)**4
        r_p = np.log10((eps / 1e-13)**2 * r)
        params = [mAp,r_p]
        dQ = scaling * 10**DP_data_dQ_dr_13_LS220_interp(params)
    except:
        # try reference eps = 1e-13
        scaling = (eps / 1e-11)**4
        r_p = np.log10((eps / 1e-11)**2 * r)
        params = [mAp,r_p]
        dQ = scaling * 10**DP_data_dQ_dr_11_LS220_interp(params)
    return dQ

def dQ_drdE_LS220(eps,mAp,r,EL):
    mAp = np.log10(mAp)
    EL = np.log10(EL)
    try:
        # try reference eps = 1e-13 first
        scaling = (eps / 1e-13)**4
        r_p = np.log10((eps / 1e-13)**2 * r)
        params = [mAp,r_p,EL]
        dQ = scaling * 10**DP_data_dQ_drdE_13_LS220_interp(params)
    except:
        # try reference eps = 1e-11
        scaling = (eps / 1e-11)**4
        r_p = np.log10((eps / 1e-11)**2 * r)
        params = [mAp,r_p,EL]
        dQ = scaling * 10**DP_data_dQ_drdE_11_LS220_interp(params)
    return dQ

def dQ_dr(eps,mAp,r,SN_profile='LS220'):
    """
    Get the dQ/dr at r

    In
    ---
    eps : scalar
        Kinetic mixing
    mAp : scalar
        Dark photon mass, MeV
    r : scalar
        CSM layer r, cm
    SN_profile : str
        Name of the SN, only LS220, TF and SFHo

    Out
    ---
    out : scalar
        The dQ/dr, erg/cm
    """
    if SN_profile == 'LS220':
        return dQ_dr_LS220(eps,mAp,r)
    elif SN_profile == 'TF':
        pass
    elif SN_profile == 'SFHo':
        pass
    else:
        raise ValueError('Option \'SN_name\' must be one of LS220, TF and SFHo')

def dQ_drdE(eps,mAp,r,EL,SN_profile='LS220'):
    """
    Get the dQ/dr/dE at r

    In
    ---
    eps : scalar
        Kinetic mixing
    mAp : scalar
        Dark photon mass, MeV
    r : scalar
        CSM layer r, cm
    EL : scalar
        Electron energy, MeV
    SN_profile : str
        Name of the SN, only LS220, TF and SFHo

    Out
    ---
    out : scalar
        The dQ/dr/dE, erg/cm/MeV
    """
    if SN_profile == 'LS220':
        return dQ_drdE_LS220(eps,mAp,r,EL)
    elif SN_profile == 'TF':
        pass
    elif SN_profile == 'SFHo':
        pass
    else:
        raise ValueError('Option \'SN_name\' must be one of LS220, TF and SFHo')
    
def radiative_loss_function(T):
    """
    Get the radiative cooling coefficient

    In
    ---
    T : scalar
        Temperature, K

    Out
    ---
    out : scalar
        The cooling coefficient, erg cm^3 / s
    """
    T = np.atleast_1d(T)
    logT = np.log10(T)
    
    # 1. Define conditions
    conds = [
        (logT > 1.0)     & (logT <= 1.422),
        (logT > 1.422)   & (logT <= 2.806),
        (logT > 2.806)   & (logT <= 3.980),
        (logT > 3.980)   & (logT <= 4.177),
        (logT > 4.177)   & (logT <= 4.443),
        (logT > 4.443)   & (logT <= 4.832),
        (logT > 4.832)   & (logT <= 5.397),
        (logT > 5.397)   & (logT <= 5.570),
        (logT > 5.570)   & (logT <= 5.890),
        (logT > 5.890)   & (logT <= 6.232),
        (logT > 6.232)   & (logT <= 6.505),
        (logT > 6.505)   & (logT <= 6.941),
        (logT > 6.941)   & (logT <= 7.385),
        (logT > 7.385)   & (logT <= 8.160)
    ]
    
    # 2. Define calculations for each condition
    # Using T directly here works because np.select only picks the correct result
    choices = [
        10**(-35.314) * T**5.452,
        10**(-29.195) * T**1.150,
        10**(-26.912) * T**0.337,
        10**(-108.273) * T**20.777,
        10**(-18.971) * T**(-0.602),
        10**(-32.195) * T**2.374,
        10**(-21.217) * T**0.102,
        10**(-0.247) * T**(-3.784),
        10**(-15.415) * T**(-1.061),
        10**(-19.275) * T**(-0.406),
        10**(-9.387) * T**(-1.992),
        10**(-22.476) * T**0.020,
        10**(-17.437) * T**(-0.706),
        10**(-25.026) * T**0.321
    ]
    
    # 3. Apply selection. default=np.nan ensures out-of-bounds doesn't return 0
    result = np.select(conds, choices, default=np.nan)
    
    # Scalar behavior for your error check
    if T.size == 1:
        if np.isnan(result[0]):
            raise ValueError(f"Temperature T={T[0]} is out of bounds")
        return result[0]
        
    return result
    # if 10 < T <= 10**1.422:
    #     return 10**(-35.314) * T**5.452
    # elif 10**1.422 < T <= 10**2.806:
    #     return 10**(-29.195) * T**1.150
    # elif 10**2.806 < T <= 10**3.980:
    #     return 10**(-26.912) * T**0.337
    # elif 10**3.980 < T <= 10**4.177:
    #     return 10**(-108.273) * T**20.777
    # elif 10**4.177 < T <= 10**4.443:
    #     return 10**(-18.971) * T**(-0.602)
    # elif 10**4.443 < T <= 10**4.832:
    #     return 10**(-32.195) * T**2.374
    # elif 10**4.832 < T <= 10**5.397:
    #     return 10**(-21.217) * T**0.102
    # elif 10**5.397 < T <= 10**5.570:
    #     return 10**(-0.247) * T**(-3.784)
    # elif 10**5.570 < T <= 10**5.890:
    #     return 10**(-15.415) * T**(-1.061)
    # elif 10**5.890 < T <= 10**6.232:
    #     return 10**(-19.275) * T**(-0.406)
    # elif 10**6.232 < T <= 10**6.505:
    #     return 10**(-9.387) * T**(-1.992)
    # elif 10**6.505 < T <= 10**6.941:
    #     return 10**(-22.476) * T**0.020
    # elif 10**6.941 < T <= 10**7.385:
    #     return 10**(-17.437) * T**(-0.706)
    # elif 10**7.385 < T <= 10**8.160:
    #     return 10**(-25.026) * T**0.321
    # else:
    #     raise ValueError("T is out of bounds")
    # #return 2e-26 * (1e7 * np.exp(-1.184e5 / (T + 1000)) + 1.4e-2 * np.sqrt(T) * np.exp(-92 / T))

def CSM_radiative_cooling_rate(r,T,X=0.7,SN_name='SN 2023ixf'):
    """
    Get the cooling coefficient

    In
    ---
    r : scalar
        CSM layer, r
    T : scalar
        CSM temeprature at layer r, K
    X : scalar
        Hydrogen mass ratio, default 0.7
    SN_name : str
        Name of the SN

    Out
    ---
    out : scalar
        The cooling rate at particular layer, erg / cm^3 / s
    """
    rho = CSM_density(r,SN_name) # CSM density at r
    nH = X * rho / mH # hydrogen number density
    Lamda = radiative_loss_function(T) * nH**2
    return Lamda

def electron_CSDA(EL,target='H'):
    """
    Electron CSDA range in hydrogen and helium

    Parameters
    ----------
    EL : scalar
        Electron kinetic energy, MeV (valid from 0.01 MeV to 1000 MeV)
    target : str
        Target that the electron is interacting with, 'H' or 'He'

    Returns
    -------
    out : scalar
        Electron CSDA range, g/cm^2
    """
    logEL = np.log10(EL)
    if target == 'H':
        sp = 10**(CSDA_e_H(logEL))
    elif target == 'He':
        sp = 10**(CSDA_e_He(logEL))
    else:
        raise ValueError('\'target\' must be either \'H\' or \'He\'.')
    return sp

def CSM_electron_stopping_length(r,EL,SN_name='SN 2023ixf'):
    """
    Electron stopping length in the layer of CSM

    Parameters
    ----------
    r : scalar
        CSM layer, r
    EL : scalar
        Electron kinetic energy, MeV (valid from 0.01 MeV to 1000 MeV)
    SN_name : str
        Name of the SN

    Returns
    -------
    out : scalar
        Electron stopping length at layer r, cm
    """
    rho = CSM_density(r,SN_name) # g/cm^3
    inv_R_H = 1/electron_CSDA(EL,target='H') # g/cm^2
    inv_R_He = 1/electron_CSDA(EL,target='He') # g/cm^3
    inv_R_mix = 0.7 * inv_R_H + 0.3 * inv_R_He
    # get stopping length
    L = 1 / inv_R_mix / rho
    return L

# def average_stopping_time(r,mAp,SN_profile='LS220'):
#     """
#     Electron stopping length in the layer of CSM

#     Parameters
#     ----------
#     r : scalar
#         CSM layer, r
#     EL : scalar
#         Electron kinetic energy, MeV (valid from 0.01 MeV to 1000 MeV)
#     SN_name : str
#         Name of the SN

#     Returns
#     -------
#     out : scalar
#         Electron stopping length at layer r, cm
#     """
#     logr = np.log10(r)
#     logmAp = np.log10(mAp)
#     if SN_profile == 'LS220':
#         avg_s_t = 10**avg_stop_time_LS220_interp((logmAp,logr))
#     return avg_s_t

def Saha_K(T, chi_ev, Z_low, Z_high):
    """Saha constant K_A,i(T) based on image formula."""
    therm_lambda_factor = (ME * KB * T / (2 * np.pi * HBAR**2))**1.5
    exp_factor = np.exp(-(chi_ev * EV_TO_ERG) / (KB * T))
    return therm_lambda_factor * (2 * Z_high / Z_low) * exp_factor

# def get_n(T, rho):
#     elements = {
#         #'H':  {'m': 1.67e-24, 'X': 1, 'chis': [13.598], 'Zs': [2, 1]},
#         'H':  {'m': 1.67e-24, 'X': 0.70, 'chis': [13.598], 'Zs': [2, 1]},
#         'He': {'m': 6.64e-24, 'X': 0.29, 'chis': [24.587, 54.417], 'Zs': [1, 2, 1]},
#         'C':  {'m': 1.99e-23, 'X': 0.003, 'chis': [11.26, 24.38, 47.89], 'Zs': [1, 1, 1, 1]},
#         'N':  {'m': 2.32e-23, 'X': 0.001, 'chis': [14.53, 29.60, 47.45], 'Zs': [1, 1, 1, 1]},
#         'O':  {'m': 2.65e-23, 'X': 0.003, 'chis': [13.62, 35.12, 54.93], 'Zs': [1, 1, 1, 1]},
#         'Fe': {'m': 9.27e-23, 'X': 0.001, 'chis': [7.90, 16.19, 30.65],  'Zs': [1, 1, 1, 1]}
#     }
    
#     n_A = {name: (data['X'] * rho / data['m']) for name, data in elements.items()}
#     K_map = {name: [Saha_K(T, chi, data['Zs'][i], data['Zs'][i+1]) 
#                     for i, chi in enumerate(data['chis'])] for name, data in elements.items()}

#     # Newton-Raphson for n_e
#     ne = 0.5 * n_A['H'] # Initial guess
#     for _ in range(15):
#         f = ne
#         df_dne = 1.0
        
#         for name, K_list in K_map.items():
#             Z = len(K_list)
#             # terms[k] = (K0*K1...K_{k-1}) * ne^(Z-k)
#             terms = [ne**Z]
#             for k in range(Z):
#                 terms.append(terms[-1] * K_list[k] / ne)
            
#             denom = sum(terms)
#             # electron contribution n_e,A = n_A * sum(k * term_k) / denom
#             num_e = sum(k * terms[k] for k in range(1, Z + 1))
            
#             # Derivatives for Newton step
#             d_terms = [Z * ne**(Z-1) if Z > 0 else 0]
#             for k in range(1, Z + 1):
#                 d_terms.append( (Z-k) * terms[k] / ne if ne != 0 else 0 )
            
#             d_num_e = sum(k * d_terms[k] for k in range(1, Z + 1))
#             d_denom = sum(d_terms)
            
#             f -= n_A[name] * (num_e / denom)
#             df_dne -= n_A[name] * (d_num_e * denom - num_e * d_denom) / (denom**2)
            
#         ne = ne - f / df_dne
        
#     return ne, n_A, K_map

# def u_gas(T, rho):
#     """
#     erg/cm^3
#     """
#     # Unpack all three outputs from get_n_total
#     ne, n_A, K_map = get_n(T, rho)
    
#     # 1. Kinetic Energy
#     n_nuclei = sum(n_A.values())
#     u_kinetic = 1.5 * (n_nuclei + ne) * KB * T
    
#     # 2. Ionization Energy
#     u_ionization = 0.0
#     for name, K_list in K_map.items():
#         Z = len(K_list)
#         # Using the same polynomial logic as get_n_total
#         terms = [ne**Z]
#         for k in range(Z):
#             terms.append(terms[-1] * K_list[k] / ne)
            
#         denom = sum(terms)
#         n_A_neutral = n_A[name] / denom
        
#         # Get ionization potentials for this element
#         # chis: [chi_0_to_1, chi_1_to_2, ...]
#         chis = [13.598, 24.587, 54.417] # Simplified lookup logic
#         # For actual code, pull from the elements dict used in get_n_total
        
#         cumul_chi = 0.0
#         for k in range(1, Z + 1):
#             # Energy to reach stage k from neutral (sum of potentials)
#             # This is derived from the "chis" list for that specific element
#             # Here we assume chis is retrieved correctly per element
#             element_chis = [13.598] if name=='H' else [24.587, 54.417] # etc.
            
#             # Sum up potentials to stage k
#             cumul_chi = sum(element_chis[:k]) * EV_TO_ERG
            
#             # n_A,k = n_A * term_k / denom
#             n_Ak = n_A[name] * terms[k] / denom
#             u_ionization += n_Ak * cumul_chi
                
#     return u_kinetic + u_ionization

# def get_n(T, rho):
#     elements = {
#         'H':  {'m': 1.67e-24, 'X': 0.70, 'chis': [13.598], 'Zs': [2, 1]},
#         'He': {'m': 6.64e-24, 'X': 0.29, 'chis': [24.587, 54.417], 'Zs': [1, 2, 1]},
#         'C':  {'m': 1.99e-23, 'X': 0.003, 'chis': [11.26, 24.38, 47.89], 'Zs': [1, 1, 1, 1]},
#         'N':  {'m': 2.32e-23, 'X': 0.001, 'chis': [14.53, 29.60, 47.45], 'Zs': [1, 1, 1, 1]},
#         'O':  {'m': 2.65e-23, 'X': 0.003, 'chis': [13.62, 35.12, 54.93], 'Zs': [1, 1, 1, 1]},
#         'Fe': {'m': 9.27e-23, 'X': 0.001, 'chis': [7.90, 16.19, 30.65],  'Zs': [1, 1, 1, 1]}
#     }
    
#     n_A = {name: (data['X'] * rho / data['m']) for name, data in elements.items()}
#     K_map = {name: [Saha_K(T, chi, data['Zs'][i], data['Zs'][i+1]) 
#                     for i, chi in enumerate(data['chis'])] for name, data in elements.items()}

#     def net_charge(ne):
#         curr_ne = max(ne, 1e-100)
#         f = curr_ne
#         for name, K_list in K_map.items():
#             Z = len(K_list)
#             # Use Logs to prevent overflow
#             log_phi = np.zeros(Z + 1)
#             for k in range(Z):
#                 log_phi[k+1] = log_phi[k] + np.log(K_list[k]) - np.log(curr_ne)
            
#             # Normalize so the largest value is 1.0 (exp(0))
#             max_log = np.max(log_phi)
#             phi = np.exp(log_phi - max_log)
            
#             denom = np.sum(phi)
#             num_e = np.sum(np.arange(Z + 1) * phi)
#             f -= n_A[name] * (num_e / denom)
#         return f

#     total_n_A = sum(n_A.values())
#     upper_bound = total_n_A * 30 + 1e-5
    
#     # This try/except ensures the code never crashes even if physics is at an extreme
#     try:
#         ne = brentq(net_charge, 1e-100, upper_bound)
#     except ValueError:
#         ne = 1e-100
        
#     return ne, n_A, K_map

# # def get_n(T, rho):
# #     elements = {
# #         'H':  {'m': 1.67e-24, 'X': 0.70, 'chis': [13.598], 'Zs': [2, 1]},
# #         'He': {'m': 6.64e-24, 'X': 0.29, 'chis': [24.587, 54.417], 'Zs': [1, 2, 1]},
# #         'C':  {'m': 1.99e-23, 'X': 0.003, 'chis': [11.26, 24.38, 47.89], 'Zs': [1, 1, 1, 1]},
# #         'N':  {'m': 2.32e-23, 'X': 0.001, 'chis': [14.53, 29.60, 47.45], 'Zs': [1, 1, 1, 1]},
# #         'O':  {'m': 2.65e-23, 'X': 0.003, 'chis': [13.62, 35.12, 54.93], 'Zs': [1, 1, 1, 1]},
# #         'Fe': {'m': 9.27e-23, 'X': 0.001, 'chis': [7.90, 16.19, 30.65],  'Zs': [1, 1, 1, 1]}
# #     }
    
# #     n_A = {name: (data['X'] * rho / data['m']) for name, data in elements.items()}
# #     K_map = {name: [Saha_K(T, chi, data['Zs'][i], data['Zs'][i+1]) 
# #                     for i, chi in enumerate(data['chis'])] for name, data in elements.items()}

# #     def net_charge(ne):
# #         # Prevent division by zero or overly small numbers that cause NaN
# #         curr_ne = max(ne, 1e-60)
# #         f = curr_ne
# #         for name, K_list in K_map.items():
# #             Z = len(K_list)
# #             terms = [1.0]
# #             for k in range(Z):
# #                 # If K/ne is huge, the stage is fully ionized; we cap it to avoid NaN
# #                 ratio = K_list[k] / curr_ne
# #                 if ratio > 1e100: 
# #                     # If ratio is massive, essentially all atoms of this element are ionized
# #                     # We use a large number that won't overflow when multiplied
# #                     new_term = 1e100
# #                 else:
# #                     new_term = terms[-1] * ratio
# #                 terms.append(new_term)
            
# #             denom = sum(terms)
# #             num_e = sum(k * terms[k] for k in range(1, Z + 1))
# #             f -= n_A[name] * (num_e / denom)
# #         return f

# #     total_n_A = sum(n_A.values())
# #     upper_bound = total_n_A * 30 + 1e-5
    
# #     # Use a slightly more stable lower bound. 1e-40 is essentially zero 
# #     # for these astrophysical densities but much safer for the solver.
# #     ne = brentq(net_charge, 1e-40, upper_bound)
        
# #     return ne, n_A, K_map

# def u_gas(T, rho):
#     """
#     The energy density stored in the gas for a given temperature and density

#     In
#     ---
#     T : scalar
#         Gas temperature, K
#     rho : scalar
#         Gas density, g/cm^3

#     Out
#     ---
#     out : scalar
#         The energy density stored in the gas, erg/cm^3
#     """
#     ne, n_A, K_map = get_n(T, rho)
    
#     elements_data = {
#         'H': [13.598], 'He': [24.587, 54.417], 'C': [11.26, 24.38, 47.89],
#         'N': [14.53, 29.60, 47.45], 'O': [13.62, 35.12, 54.93], 'Fe': [7.90, 16.19, 30.65]
#     }
    
#     n_nuclei = sum(n_A.values())
#     u_kinetic = 1.5 * (n_nuclei + ne) * KB * T
    
#     u_ionization = 0.0
#     curr_ne = max(ne, 1e-100) # Safety
#     for name, K_list in K_map.items():
#         Z = len(K_list)
#         # Apply the same Log-Normalization logic here
#         log_phi = np.zeros(Z + 1)
#         for k in range(Z):
#             log_phi[k+1] = log_phi[k] + np.log(K_list[k]) - np.log(curr_ne)
        
#         max_log = np.max(log_phi)
#         phi = np.exp(log_phi - max_log)
#         denom = np.sum(phi)
        
#         element_chis = elements_data[name]
#         for k in range(1, Z + 1):
#             cumul_chi = sum(element_chis[:k]) * EV_TO_ERG
#             # n_Ak = total element density * fraction in stage k
#             n_Ak = n_A[name] * (phi[k] / denom)
#             u_ionization += n_Ak * cumul_chi
                
#     return u_kinetic + u_ionization

def get_n(T, rho):
    elements = {
        'H':  {'m': 1.67e-24, 'X': 0.70, 'chis': [13.598], 'Zs': [2, 1]},
        'He': {'m': 6.64e-24, 'X': 0.29, 'chis': [24.587, 54.417], 'Zs': [1, 2, 1]},
        'C':  {'m': 1.99e-23, 'X': 0.003, 'chis': [11.26, 24.38, 47.89], 'Zs': [1, 1, 1, 1]},
        'N':  {'m': 2.32e-23, 'X': 0.001, 'chis': [14.53, 29.60, 47.45], 'Zs': [1, 1, 1, 1]},
        'O':  {'m': 2.65e-23, 'X': 0.003, 'chis': [13.62, 35.12, 54.93], 'Zs': [1, 1, 1, 1]},
        'Fe': {'m': 9.27e-23, 'X': 0.001, 'chis': [7.90, 16.19, 30.65],  'Zs': [1, 1, 1, 1]}
    }
    
    n_A = {name: (data['X'] * rho / data['m']) for name, data in elements.items()}
    # Safety: ensure T isn't so low that Saha_K returns 0.0
    K_map = {name: [max(Saha_K(T, chi, data['Zs'][i], data['Zs'][i+1]), 1e-300) 
                    for i, chi in enumerate(data['chis'])] for name, data in elements.items()}

    def net_charge(ne):
        curr_ne = max(ne, 1e-100)
        f = curr_ne
        for name, K_list in K_map.items():
            Z = len(K_list)
            # log_phi[k] is the log of the abundance of stage k relative to stage 0
            log_phi = np.zeros(Z + 1)
            for k in range(Z):
                log_phi[k+1] = log_phi[k] + np.log(K_list[k]) - np.log(curr_ne)
            
            # Stabilization: subtract max to prevent overflow in exp()
            max_log = np.max(log_phi)
            phi = np.exp(log_phi - max_log)
            denom = np.sum(phi)
            
            # Mean ionization stage <Z> = sum(k * n_k) / sum(n_k)
            mean_z = np.sum(np.arange(Z + 1) * phi) / denom
            f -= n_A[name] * mean_z
        return f

    total_n_A = sum(n_A.values())
    upper_bound = total_n_A * 30 + 1e-5
    
    # Verify signs to avoid brentq error
    try:
        f_min = net_charge(1e-100)
        f_max = net_charge(upper_bound)
        if f_min * f_max > 0:
            return (1e-100 if f_min > 0 else upper_bound), n_A, K_map
        return brentq(net_charge, 1e-100, upper_bound), n_A, K_map
    except:
        return 1e-100, n_A, K_map

def u_gas(T, rho):
    ne, n_A, K_map = get_n(T, rho)
    elements_data = {
        'H': [13.598], 'He': [24.587, 54.417], 'C': [11.26, 24.38, 47.89],
        'N': [14.53, 29.60, 47.45], 'O': [13.62, 35.12, 54.93], 'Fe': [7.90, 16.19, 30.65]
    }
    
    u_kinetic = 1.5 * (sum(n_A.values()) + ne) * KB * T
    u_ionization = 0.0
    curr_ne = max(ne, 1e-100)

    for name, K_list in K_map.items():
        Z = len(K_list)
        # Re-calculate stable fractions
        log_phi = np.zeros(Z + 1)
        for k in range(Z):
            log_phi[k+1] = log_phi[k] + np.log(K_list[k]) - np.log(curr_ne)
        
        max_log = np.max(log_phi)
        phi = np.exp(log_phi - max_log)
        denom = np.sum(phi)
        
        element_chis = elements_data[name]
        for k in range(1, Z + 1):
            cumul_chi = sum(element_chis[:k]) * EV_TO_ERG
            # stage_fraction = phi[k] / denom
            u_ionization += n_A[name] * (phi[k] / denom) * cumul_chi
                
    return u_kinetic + u_ionization

# def u_gas(T, rho):
#     """
#     The energy density stored in the gas for a given temperature and density

#     In
#     ---
#     T : scalar
#         Gas temperature, K
#     rho : scalar
#         Gas density, g/cm^3

#     Out
#     ---
#     out : scalar
#         The energy density stored in the gas, erg/cm^3
#     """
#     ne, n_A, K_map = get_n(T, rho)
    
#     elements_data = {
#         'H': [13.598], 'He': [24.587, 54.417], 'C': [11.26, 24.38, 47.89],
#         'N': [14.53, 29.60, 47.45], 'O': [13.62, 35.12, 54.93], 'Fe': [7.90, 16.19, 30.65]
#     }
    
#     n_nuclei = sum(n_A.values())
#     u_kinetic = 1.5 * (n_nuclei + ne) * KB * T
    
#     u_ionization = 0.0
#     for name, K_list in K_map.items():
#         Z = len(K_list)
#         terms = [1.0]
#         for k in range(Z):
#             terms.append(terms[-1] * K_list[k] / ne)
            
#         denom = sum(terms)
#         element_chis = elements_data[name]
        
#         for k in range(1, Z + 1):
#             cumul_chi = sum(element_chis[:k]) * EV_TO_ERG
#             n_Ak = n_A[name] * (terms[k] / denom)
#             u_ionization += n_Ak * cumul_chi
                
#     return u_kinetic + u_ionization

def u_rad(T):
    """
    The energy density stored in the radiation for a given temperature, assuming LTE

    In
    ---
    T : scalar
        Temperature, K
    
    Out
    ---
    out : scalar
        The stored energy density, erg/cm^3
    """
    u_rad = a * T**4 
    return u_rad

def u_total(r,T,include_rad=True,SN_name='SN 2023ixf'):
    """
    The energy density stored in CSM at layer r

    In
    ---
    r : scalar
        CSM layer r, cm
    T : scalar
        CSM temperature, K
    include_rad : bool
        Whether the contribution from radiation is included, True assume radiation + gas is in LTE
    SN_name : str
        Name of the SN

    Out
    ---
    out : scalar
        The stored energy density in a layer of CSM, erg/cm^3
    """
    rho = CSM_density(r,SN_name)
    if include_rad is True:
        return u_rad(T) + u_gas(T, rho)
    else:
        return u_gas(T, rho)

def average_stopping_time(r,eps,mAp,SN_name='SN 2023ixf',SN_profile='LS220'):
    """
    Average stopping time for electron at CSM layer r

    In
    ---
    r : scalar
        CSM radius, cm
    eps : scalar
        Kinetic mixing parameter
    mAp : scalar
        Dark photon mass, MeV
    SN_name : str
        Name of the SN
    SN_profile : str
        Name of the SN explosion profile

    Out
    ---
    out : scalar
        The average stopping time for electron, s
    """
    # Stopping time weighted by the EL distribution at layer r, s erg / cm / MeV
    EL_weighted_time = quad(lambda EL: dQ_drdE(eps,mAp,r,EL,SN_profile)*CSM_electron_stopping_length(r,EL,SN_name),1,501)[0] / c
    # Total energy will be deposited, s
    Q = dQ_dr(eps,mAp,r,SN_profile) #quad(lambda EL: dQ_drdE(eps,mAp,r,EL),1,501)[0]
    return (EL_weighted_time / Q)[0]

def dQ_dV(r,eps,mAp,SN_name='SN 2023ixf',SN_profile='LS220'):
    """
    DP dQ/dV at CSM layer r

    In
    ---
    r : scalar
        CSM radius, cm
    eps : scalar
        Kinetic mixing parameter
    mAp : scalar
        Dark photon mass, MeV
    SN_name : str
        Name of the SN
    SN_profile : str
        Name of the SN explosion profile

    Out
    ---
    out : scalar
        dQ/dV at layer r, erg/cm^3
    """
    # dQ/dV at layer r
    return dQ_dr(eps,mAp,r,SN_profile) / 4 / np.pi / r**2

def get_T(r,eps,mAp,include_rad=True,SN_name='SN 2023ixf',SN_profile='LS220',x0=5000,x1=1e5,xtol=None,rtol=None):
    """
    Get the temperature of CSM at layer r due to DP energy deposition

    In
    ---
    r : scalar
        CSM radius, cm
    eps : scalar
        Kinetic mixing parameter
    mAp : scalar
        Dark photon mass, MeV
    include_rad : bool
        Included radition, if True, assuming radiation and gas are in LTE
        if False, it's non-LTE, radiation will not contribute to the storage of energy
    SN_name : str
        Name of the SN
    SN_profile : str
        Name of the SN explosion profile

    Out
    ---
    out : tup
        A tuple of (dQ/dV in erg/cm^3, flag, error %)
        If the flag is False, then the root finding algorithm fails to find the T
    """
    dQ = dQ_dV(r,eps,mAp,SN_name,SN_profile)[0] # dQ/dV deposited at layer r, erg/cm^3
    def _f(T):
        return dQ - u_total(r,T,include_rad,SN_name)
    
    # def _f_with_derivatives(x, h=1e-5):
    #     # We use a larger h than the 1e-20 used for first-derivatives 
    #     # to avoid cancellation in the 2nd derivative calculation.
    #     z = x + 1j * h
    #     fz = _f(z)
    #     fx = _f(x) # We need the actual real evaluation for f''
        
    #     val = fx
    #     # 1st derivative (Complex-step)
    #     f_prime = fz.imag / h
    #     # 2nd derivative (Real-part difference)
    #     f_double_prime = 2 * (fx - fz.real) / (h**2)
        
    #     return val, f_prime, f_double_prime
    #sol = root_scalar(_f_with_derivatives, fprime=True,fprime2=True,x0=x0,x1=x1, method='halley',xtol=xtol,rtol=rtol)
    sol = root_scalar(_f,x0=x0, x1=x1, method='secant',xtol=xtol,rtol=rtol)
    
    if np.isnan(sol.root):
        # secant method fails to converge, using a different way
        new_x0 = root_scalar(_f,x0=x0,xtol=xtol,rtol=rtol).root
        def _f_with_derivatives(x, h=1e-20):
            # We use a larger h than the 1e-20 used for first-derivatives 
            # to avoid cancellation in the 2nd derivative calculation.
            z = x + 1j * h
            fz = _f(z)
            fx = _f(x) # We need the actual real evaluation for f''
            
            val = fx
            # 1st derivative (Complex-step)
            f_prime = fz.imag / h
            # 2nd derivative (Real-part difference)
            #f_double_prime = 2 * (fx - fz.real) / (h**2)
            
            return val, f_prime#, f_double_prime
        sol = root_scalar(_f_with_derivatives, fprime=True,x0=new_x0, method='newton',xtol=xtol,rtol=rtol)

    return sol.root,sol.converged,np.abs(_f(sol.root)/dQ) * 100