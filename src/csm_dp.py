import numpy as np
from scipy.interpolate import RegularGridInterpolator,interp1d
from scipy.integrate import quad
from scipy.optimize import brentq,root_scalar,newton
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
rho_SN2023ixf_sharp_data = np.log10(np.loadtxt(BASE_DIR / 'density/SN2023ixf_Galan.txt').T)
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

# --- Internal energy --- #
SN2023ixf_logu_gas = np.loadtxt(BASE_DIR / 'internal_energy/u_gas_only.txt')
SN2023ixf_logu_tot = np.loadtxt(BASE_DIR / 'internal_energy/u_total.txt')

SN2023ixf_s_logu_gas = np.loadtxt(BASE_DIR / 'internal_energy/u_gas_only_SN 2023ixf s.txt')
SN2023ixf_s_logu_tot = np.loadtxt(BASE_DIR / 'internal_energy/u_total_SN 2023ixf s.txt')

logT_axis = np.log10(np.logspace(0,15,600))
logR_axis = np.log10(np.logspace(np.log10(5e13),np.log10(8e15),300))

SN2023ixf_LS220_u_gas_interp = RegularGridInterpolator((logR_axis,logT_axis),SN2023ixf_logu_gas,method='linear', bounds_error=False, fill_value=np.nan)
SN2023ixf_LS220_u_tot_interp = RegularGridInterpolator((logR_axis,logT_axis),SN2023ixf_logu_tot,method='linear', bounds_error=False, fill_value=np.nan)

SN2023ixf_s_LS220_u_gas_interp = RegularGridInterpolator((logR_axis,logT_axis),SN2023ixf_s_logu_gas,method='linear', bounds_error=False, fill_value=np.nan)
SN2023ixf_s_LS220_u_tot_interp = RegularGridInterpolator((logR_axis,logT_axis),SN2023ixf_s_logu_tot,method='linear', bounds_error=False, fill_value=np.nan)

# --- Opacity --- #
# Low T opacity table from Ferguson et al, T < 10000 K
raw_Ferguson = np.loadtxt(BASE_DIR / 'opacity/Ferguson.txt')
logOpacity_Ferguson = raw_Ferguson[1:,1:]
logR_Ferguson = raw_Ferguson[0,1:]
logT_Ferguson = raw_Ferguson[1:,0]
logOpa_Ferguson_interp = RegularGridInterpolator((logT_Ferguson, logR_Ferguson),logOpacity_Ferguson,method='linear', bounds_error=False, fill_value=np.nan)  # (logT,logR)

# Low T opacity table from Ferguson et al, T < 10000 K
raw_OPAL = np.loadtxt(BASE_DIR / 'opacity/OPAL.txt')
logOpacity_OPAL = raw_OPAL[1:,1:]
logR_OPAL = raw_OPAL[0,1:]
logT_OPAL = raw_OPAL[1:,0]
logOpa_OPAL_interp = RegularGridInterpolator((logT_OPAL, logR_OPAL),logOpacity_OPAL,method='linear', bounds_error=False, fill_value=np.nan) # (logT,logR)                                         method='linear', 

# --- Electron CSDA --- #
# load CSDA data
CSDA_electron = np.loadtxt(BASE_DIR / 'CSDA/electron.txt',skiprows=6).T
# interpolate, in log-scale
CSDA_e_H = interp1d(np.log10(CSDA_electron[0]),np.log10(CSDA_electron[-2])) # hydrogen
CSDA_e_He = interp1d(np.log10(CSDA_electron[0]),np.log10(CSDA_electron[-1])) # helium

# --- Electron stopping power --- $
stop_power_e_H = np.loadtxt(BASE_DIR / 'stopping_power/electron_H.txt',skiprows=8).T
stop_power_e_He = np.loadtxt(BASE_DIR / 'stopping_power/electron_He.txt',skiprows=8).T
# interpolate, in log-scale
stop_power_e_H_interp = interp1d(np.log10(stop_power_e_H[0]),np.log10(stop_power_e_H[-1]))
stop_power_e_He_interp = interp1d(np.log10(stop_power_e_He[0]),np.log10(stop_power_e_He[-1]))

# # --- Average stopping time and efficiency --- #
avg_stop_time_eff_LS220 = np.loadtxt(BASE_DIR / 'average_stopping_time/CSM_average_stoptime_efficiency_LS220.txt')
avg_stop_time_eff_SFHo = np.loadtxt(BASE_DIR / 'average_stopping_time/CSM_average_stoptime_efficiency_SFHo.txt')
avg_stop_time_eff_TF = np.loadtxt(BASE_DIR / 'average_stopping_time/CSM_average_stoptime_efficiency_TF.txt')
avg_stop_time_eff_LS220_s = np.loadtxt(BASE_DIR / 'average_stopping_time/CSM_average_stoptime_efficiency_LS220_SN 2023ixf s.txt')


logStopTime_LS220 = np.log10(avg_stop_time_eff_LS220[:, 3].reshape(7,50,35))
logStopTime_SFHo = np.log10(avg_stop_time_eff_SFHo[:, 3].reshape(7,50,35))
logStopTime_TF = np.log10(avg_stop_time_eff_TF[:, 3].reshape(7,50,35))
logStopTime_LS220_s = np.log10(avg_stop_time_eff_LS220_s[:, 3].reshape(7,50,35))

AvgEfficiency_LS220 = avg_stop_time_eff_LS220[:, 4].reshape(7,50,35)
AvgEfficiency_LS220[AvgEfficiency_LS220  < 1e-50] = 1e-50 # some efficiency could be close to 0 and cannot be taken log10
logEfficiency_LS220 = np.log10(AvgEfficiency_LS220)

AvgEfficiency_SFHo = avg_stop_time_eff_SFHo[:, 4].reshape(7,50,35)
AvgEfficiency_SFHo[AvgEfficiency_SFHo  < 1e-50] = 1e-50 # some efficiency could be close to 0 and cannot be taken log10
logEfficiency_SFHo = np.log10(AvgEfficiency_SFHo)

AvgEfficiency_TF = avg_stop_time_eff_TF[:, 4].reshape(7,50,35)
AvgEfficiency_TF[AvgEfficiency_TF  < 1e-50] = 1e-50 # some efficiency could be close to 0 and cannot be taken log10
logEfficiency_TF = np.log10(AvgEfficiency_TF)

AvgEfficiency_LS220_s = avg_stop_time_eff_LS220_s[:, 4].reshape(7,50,35)
AvgEfficiency_LS220_s[AvgEfficiency_LS220_s  < 1e-50] = 1e-50 # some efficiency could be close to 0 and cannot be taken log10
logEfficiency_LS220_s = np.log10(AvgEfficiency_LS220_s)

# define axis
logeps_axis = np.log10([1.5e-14,3.3e-14,1e-13,3.3e-13,1e-12,3.3e-12,1e-11])
logmAp_axis = np.log10(np.logspace(2.99922e-02,2.7,50))
logr_axis = np.log10(np.logspace(np.log10(5e13),np.log10(8e15),35))
# interpolation
AvgStopTime_LS220_interp = RegularGridInterpolator((logeps_axis,logmAp_axis,logr_axis),logStopTime_LS220,method='linear', bounds_error=False, fill_value=np.nan)
AvgEfficiency_LS220_interp = RegularGridInterpolator((logeps_axis,logmAp_axis,logr_axis),logEfficiency_LS220 ,method='linear', bounds_error=False, fill_value=np.nan)

AvgStopTime_LS220_s_interp = RegularGridInterpolator((logeps_axis,logmAp_axis,logr_axis),logStopTime_LS220_s,method='linear', bounds_error=False, fill_value=np.nan)
AvgEfficiency_LS220_s_interp = RegularGridInterpolator((logeps_axis,logmAp_axis,logr_axis),logEfficiency_LS220_s ,method='linear', bounds_error=False, fill_value=np.nan)

AvgStopTime_SFHo_interp = RegularGridInterpolator((logeps_axis,logmAp_axis,logr_axis),logStopTime_SFHo,method='linear', bounds_error=False, fill_value=np.nan)
AvgEfficiency_SFHo_interp = RegularGridInterpolator((logeps_axis,logmAp_axis,logr_axis),logEfficiency_SFHo ,method='linear', bounds_error=False, fill_value=np.nan)

AvgStopTime_TF_interp = RegularGridInterpolator((logeps_axis,logmAp_axis,logr_axis),logStopTime_TF,method='linear', bounds_error=False, fill_value=np.nan)
AvgEfficiency_TF_interp = RegularGridInterpolator((logeps_axis,logmAp_axis,logr_axis),logEfficiency_TF ,method='linear', bounds_error=False, fill_value=np.nan)

# --- Dark Photon Luminosity --- #
# * -- With EL distribution -- * #
# Load data and restructure it into (91,41,32)
DP_data_dQ_drdE_11_LS220 = np.loadtxt(BASE_DIR / 'dp_luminosity/time_integrated/LS220_dQ_drdE_eps_1e-11.txt')[:, 3].reshape(91,41,32)
DP_data_dQ_drdE_13_LS220 = np.loadtxt(BASE_DIR / 'dp_luminosity/time_integrated/LS220_dQ_drdE_eps_1e-13.txt')[:, 3].reshape(91,41,32)
DP_data_dQ_drdE_11_SFHo = np.loadtxt(BASE_DIR / 'dp_luminosity/time_integrated/SFHo_dQ_drdE_eps_1e-11.txt')[:, 3].reshape(91,41,32)
DP_data_dQ_drdE_13_SFHo = np.loadtxt(BASE_DIR / 'dp_luminosity/time_integrated/SFHo_dQ_drdE_eps_1e-13.txt')[:, 3].reshape(91,41,32)
DP_data_dQ_drdE_11_TF = np.loadtxt(BASE_DIR / 'dp_luminosity/time_integrated/TF_dQ_drdE_eps_1e-11.txt')[:, 3].reshape(91,41,32)
DP_data_dQ_drdE_13_TF = np.loadtxt(BASE_DIR / 'dp_luminosity/time_integrated/TF_dQ_drdE_eps_1e-13.txt')[:, 3].reshape(91,41,32)

mAp_axis = np.linspace(0,2.7,91)
r_axis = np.linspace(12,20,41)
EL_axis = np.linspace(0,2.7,32)

# eps = 1e-11 and 1e-13
DP_data_dQ_drdE_11_LS220_interp = RegularGridInterpolator((mAp_axis, r_axis, EL_axis),DP_data_dQ_drdE_11_LS220,method='linear',bounds_error=True)
DP_data_dQ_drdE_13_LS220_interp = RegularGridInterpolator((mAp_axis, r_axis, EL_axis),DP_data_dQ_drdE_13_LS220,method='linear',bounds_error=True)
DP_data_dQ_drdE_11_SFHo_interp = RegularGridInterpolator((mAp_axis, r_axis, EL_axis),DP_data_dQ_drdE_11_SFHo,method='linear',bounds_error=True)
DP_data_dQ_drdE_13_SFHo_interp = RegularGridInterpolator((mAp_axis, r_axis, EL_axis),DP_data_dQ_drdE_13_SFHo,method='linear',bounds_error=True)
DP_data_dQ_drdE_11_TF_interp = RegularGridInterpolator((mAp_axis, r_axis, EL_axis),DP_data_dQ_drdE_11_TF,method='linear',bounds_error=True)
DP_data_dQ_drdE_13_TF_interp = RegularGridInterpolator((mAp_axis, r_axis, EL_axis),DP_data_dQ_drdE_13_TF,method='linear',bounds_error=True)

# * -- With EL distribution integrated -- * #
# Load data and restructure it into (91,41,32)
DP_data_dQ_dr_11_LS220 = np.loadtxt(BASE_DIR / 'dp_luminosity/time_integrated/LS220_dQ_dr_eps_1e-11.txt')[:, 2].reshape(91,41)
DP_data_dQ_dr_13_LS220 = np.loadtxt(BASE_DIR / 'dp_luminosity/time_integrated/LS220_dQ_dr_eps_1e-13.txt')[:, 2].reshape(91,41)
DP_data_dQ_dr_11_SFHo = np.loadtxt(BASE_DIR / 'dp_luminosity/time_integrated/SFHo_dQ_dr_eps_1e-11.txt')[:, 2].reshape(91,41)
DP_data_dQ_dr_13_SFHo = np.loadtxt(BASE_DIR / 'dp_luminosity/time_integrated/SFHo_dQ_dr_eps_1e-13.txt')[:, 2].reshape(91,41)
DP_data_dQ_dr_11_TF = np.loadtxt(BASE_DIR / 'dp_luminosity/time_integrated/TF_dQ_dr_eps_1e-11.txt')[:, 2].reshape(91,41)
DP_data_dQ_dr_13_TF = np.loadtxt(BASE_DIR / 'dp_luminosity/time_integrated/TF_dQ_dr_eps_1e-13.txt')[:, 2].reshape(91,41)

# eps = 1e-11 and 1e-13
DP_data_dQ_dr_11_LS220_interp = RegularGridInterpolator((mAp_axis, r_axis),DP_data_dQ_dr_11_LS220,method='linear',bounds_error=True)
DP_data_dQ_dr_13_LS220_interp = RegularGridInterpolator((mAp_axis, r_axis),DP_data_dQ_dr_13_LS220,method='linear',bounds_error=True)
DP_data_dQ_dr_11_SFHo_interp = RegularGridInterpolator((mAp_axis, r_axis),DP_data_dQ_dr_11_SFHo,method='linear',bounds_error=True)
DP_data_dQ_dr_13_SFHo_interp = RegularGridInterpolator((mAp_axis, r_axis),DP_data_dQ_dr_13_SFHo,method='linear',bounds_error=True)
DP_data_dQ_dr_11_TF_interp = RegularGridInterpolator((mAp_axis, r_axis),DP_data_dQ_dr_11_TF,method='linear',bounds_error=True)
DP_data_dQ_dr_13_TF_interp = RegularGridInterpolator((mAp_axis, r_axis),DP_data_dQ_dr_13_TF,method='linear',bounds_error=True)


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

def average_stopping_time(eps,mAp,r,SN_profile='LS220'):
    logeps = np.log10(eps)
    logmAp = np.log10(mAp)
    logr = np.log10(r)
    if SN_profile == 'LS220':
        avg_time = 10**AvgStopTime_LS220_interp((logeps,logmAp,logr))
    elif SN_profile == 'LS220 s':
        avg_time = 10**AvgStopTime_LS220_s_interp((logeps,logmAp,logr))
    elif SN_profile == 'SFHo':
        avg_time = 10**AvgStopTime_SFHo_interp((logeps,logmAp,logr))
    elif SN_profile == 'TF':
        avg_time = 10**AvgStopTime_TF_interp((logeps,logmAp,logr))
    else:
        raise ValueError('\'SN_profile\' incorrect')
    return avg_time

def average_efficiency(eps,mAp,r,SN_profile='LS220'):
    logeps = np.log10(eps)
    logmAp = np.log10(mAp)
    logr = np.log10(r)
    if SN_profile == 'LS220':
        avg_eff = 10**AvgEfficiency_LS220_interp((logeps,logmAp,logr))
    elif SN_profile == 'LS220 s':
        avg_eff = 10**AvgEfficiency_LS220_s_interp((logeps,logmAp,logr))
    elif SN_profile == 'SFHo':
        avg_eff = 10**AvgEfficiency_SFHo_interp((logeps,logmAp,logr))
    elif SN_profile == 'TF':
        avg_eff = 10**AvgEfficiency_TF_interp((logeps,logmAp,logr))
    else:
        raise ValueError('\'SN_profile\' incorrect')
    return avg_eff

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
        params = (mAp,r_p)
        dQ = scaling * 10**DP_data_dQ_dr_13_LS220_interp(params)
    except:
        # try reference eps = 1e-13
        scaling = (eps / 1e-11)**4
        r_p = np.log10((eps / 1e-11)**2 * r)
        params = (mAp,r_p)
        dQ = scaling * 10**DP_data_dQ_dr_11_LS220_interp(params)
    return dQ

def dQ_dr_SFHo(eps,mAp,r):
    mAp = np.log10(mAp)
    try:
        # try reference eps = 1e-13 first
        scaling = (eps / 1e-13)**4
        r_p = np.log10((eps / 1e-13)**2 * r)
        params = (mAp,r_p)
        dQ = scaling * 10**DP_data_dQ_dr_13_SFHo_interp(params)
    except:
        # try reference eps = 1e-13
        scaling = (eps / 1e-11)**4
        r_p = np.log10((eps / 1e-11)**2 * r)
        params = (mAp,r_p)
        dQ = scaling * 10**DP_data_dQ_dr_11_SFHo_interp(params)
    return dQ

def dQ_dr_TF(eps,mAp,r):
    mAp = np.log10(mAp)
    try:
        # try reference eps = 1e-13 first
        scaling = (eps / 1e-13)**4
        r_p = np.log10((eps / 1e-13)**2 * r)
        params = (mAp,r_p)
        dQ = scaling * 10**DP_data_dQ_dr_13_TF_interp(params)
    except:
        # try reference eps = 1e-13
        scaling = (eps / 1e-11)**4
        r_p = np.log10((eps / 1e-11)**2 * r)
        params = (mAp,r_p)
        dQ = scaling * 10**DP_data_dQ_dr_11_TF_interp(params)
    return dQ

def dQ_drdE_LS220(eps,mAp,r,EL):
    mAp = np.log10(mAp)
    EL = np.log10(EL)
    try:
        # try reference eps = 1e-13 first
        scaling = (eps / 1e-13)**4
        r_p = np.log10((eps / 1e-13)**2 * r)
        params = (mAp,r_p,EL)
        dQ = scaling * 10**DP_data_dQ_drdE_13_LS220_interp(params)
    except:
        # try reference eps = 1e-11
        scaling = (eps / 1e-11)**4
        r_p = np.log10((eps / 1e-11)**2 * r)
        params = (mAp,r_p,EL)
        dQ = scaling * 10**DP_data_dQ_drdE_11_LS220_interp(params)
    return dQ

def dQ_drdE_SFHo(eps,mAp,r,EL):
    mAp = np.log10(mAp)
    EL = np.log10(EL)
    try:
        # try reference eps = 1e-13 first
        scaling = (eps / 1e-13)**4
        r_p = np.log10((eps / 1e-13)**2 * r)
        params = (mAp,r_p,EL)
        dQ = scaling * 10**DP_data_dQ_drdE_13_SFHo_interp(params)
    except:
        # try reference eps = 1e-11
        scaling = (eps / 1e-11)**4
        r_p = np.log10((eps / 1e-11)**2 * r)
        params = (mAp,r_p,EL)
        dQ = scaling * 10**DP_data_dQ_drdE_11_SFHo_interp(params)
    return dQ

def dQ_drdE_TF(eps,mAp,r,EL):
    mAp = np.log10(mAp)
    EL = np.log10(EL)
    try:
        # try reference eps = 1e-13 first
        scaling = (eps / 1e-13)**4
        r_p = np.log10((eps / 1e-13)**2 * r)
        params = (mAp,r_p,EL)
        dQ = scaling * 10**DP_data_dQ_drdE_13_TF_interp(params)
    except:
        # try reference eps = 1e-11
        scaling = (eps / 1e-11)**4
        r_p = np.log10((eps / 1e-11)**2 * r)
        params = (mAp,r_p,EL)
        dQ = scaling * 10**DP_data_dQ_drdE_11_TF_interp(params)
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
        return dQ_dr_TF(eps,mAp,r)
    elif SN_profile == 'SFHo':
        return dQ_dr_SFHo(eps,mAp,r)
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
        return dQ_drdE_TF(eps,mAp,r,EL)
    elif SN_profile == 'SFHo':
        return dQ_drdE_SFHo(eps,mAp,r,EL)
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

def stopping_power_electron(Ek,target='H'):
    """
    Stopping power for electron in different target

    Parameters
    ----------
    Ek : scalar
        Electron kinetic energy, MeV (valid from 0.01 MeV to 1000 MeV)
    target : str
        Target that the electron is interacting with, 'H' or 'He'

    Returns
    -------
    out : scalar
        Electron stopping power, MeV cm^2/g
    """
    logEk = np.log10(Ek)
    if target == 'H':
        sp = 10**(stop_power_e_H_interp(logEk))
    elif target == 'He':
        sp = 10**(stop_power_e_He_interp(logEk))
    else:
        raise ValueError('\'target\' must be either \'H\' or \'He\'.')
    return sp

def CSM_electron_stopping_length(r,EL,SN_name='SN 2023ixf',CSDA=True,E0=0.5):
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
    CSDA : bool
        Using CSDA to estimate the stopping length
    E0 : scalar
        The lowest energy that can be considered electron is stopped, MeV

    Returns
    -------
    out : scalar
        Electron stopping length at layer r, cm
    """
    rho = CSM_density(r,SN_name) # g/cm^3
    if CSDA is True:
        inv_R_H = 1 / electron_CSDA(EL,target='H') # g/cm^2
        inv_R_He = 1 / electron_CSDA(EL,target='He') # g/cm^3
        inv_R_mix = 0.7 * inv_R_H + 0.3 * inv_R_He
        # get stopping length
        L = 1 / inv_R_mix / rho
    elif CSDA is False:
        def _one_over_S_mix_rho(EL):
            S_H = stopping_power_electron(EL,target='H')
            S_He = stopping_power_electron(EL,target='He')
            return 1 / (S_H * 0.7 + S_He * 0.3) / rho
        L = quad(_one_over_S_mix_rho,EL,E0)[0]    
    else:
        raise ValueError('\'CSDA\' must be either True or False.')
    return L

def Saha_K(T, chi_ev, Z_low, Z_high):
    """Saha constant K_A,i(T) based on image formula."""
    therm_lambda_factor = (ME * KB * T / (2 * np.pi * HBAR**2))**1.5
    exp_factor = np.exp(-(chi_ev * EV_TO_ERG) / (KB * T))
    return therm_lambda_factor * (2 * Z_high / Z_low) * exp_factor

def get_n(T, rho):
    elements = {
        'H':  {'m': 1.67e-24, 'X': 0.70, 'chis': [13.598], 'Zs': [2, 1]},
        'He': {'m': 6.64e-24, 'X': 0.28, 'chis': [24.587, 54.417], 'Zs': [1, 2, 1]},
        'C':  {'m': 1.99e-23, 'X': 0.006, 'chis': [11.26, 24.38, 47.89], 'Zs': [1, 1, 1, 1]},
        'N':  {'m': 2.32e-23, 'X': 0.004, 'chis': [14.53, 29.60, 47.45], 'Zs': [1, 1, 1, 1]},
        'O':  {'m': 2.65e-23, 'X': 0.006, 'chis': [13.62, 35.12, 54.93], 'Zs': [1, 1, 1, 1]},
        'Fe': {'m': 9.27e-23, 'X': 0.004, 'chis': [7.90, 16.19, 30.65],  'Zs': [1, 1, 1, 1]}
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

def u_interp(r,T,include_rad=True,SN_name='SN 2023ixf'):
    logr = np.log10(r)
    logT = np.log10(T)
    if SN_name == 'SN 2023ixf':
        if include_rad is True:
            return 10**SN2023ixf_LS220_u_tot_interp((logr,logT))
        else:
            return 10**SN2023ixf_LS220_u_gas_interp((logr,logT))
    elif SN_name == 'SN 2023ixf s':
        if include_rad is True:
            return 10**SN2023ixf_s_LS220_u_tot_interp((logr,logT))
        else:
            return 10**SN2023ixf_s_LS220_u_gas_interp((logr,logT))
    else:
        raise ValueError('Wrong SN name')

def average_stopping_time_numerical(r,eps,mAp,SN_name='SN 2023ixf',SN_profile='LS220'):
    """
    Solving the average stopping time for electron at CSM layer r numerically

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
    return (EL_weighted_time / Q)

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

def get_T(r,eps,mAp,efficiency=1,include_rad=True,SN_name='SN 2023ixf',SN_profile='LS220',bracket=[1,1e15],xtol=None,rtol=None,interp_u=True):
    """
    Get the temperature of CSM at layer r due to DP energy deposition
    Method: Brentq searching in log-valued T space

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
    bracket : list
        The boundary for solver to find T, default is [1,1e15] K
        Info: To perform the solver, we search T in log-valued range in order to create sufficiently large slope.
              If it is perfomed in linear scale, it could cause the solver truncated at some near-flat region.
    xtol : bool
        Option for root_scalar
    rtol : bool
        Option for root_scalr
    interp_u : bool
        Using lookup table for finding internal energy, default is True. However, the table is only valid for
        temperature in [1,1e15] K and radius in [5e13,8e15] cm
        Set it False will force the algorithm to find the internal energy directly using root finding algorithm
        This will increases the computational cost.

    Out
    ---
    out : tup
        A tuple of (T, flag, error %)
        If the flag is False, then the root finding algorithm fails to find the T
    """
    #logT0 = np.log10(T0)
    dQ = dQ_dV(r,eps,mAp,SN_name,SN_profile) * efficiency # dQ/dV deposited at layer r, erg/cm^3
    
    def _f(logT):
        T = 10**logT
        if interp_u is True:
            u = u_interp(r,T,include_rad,SN_name)
        else:
            u = u_total(r,T,include_rad,SN_name)
        return dQ - u #u_total(r,T,include_rad,SN_name)
    
    log_bracket = np.log10(bracket) # initial guess on T in log10 valued
    sol = root_scalar(_f,method='brentq', bracket=log_bracket,xtol=xtol,rtol=rtol)
    sol_T = 10**sol.root
    return sol_T,sol.converged,np.abs(_f(sol.root)/dQ) * 100