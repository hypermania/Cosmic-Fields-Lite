import numpy as np
from numpy import sqrt, pi, sin, cos, log, log10, exp, tanh, sinh, cosh
import numpy.fft as fft
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.font_manager as font_manager
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy
import scipy.special as sc
import struct
import re
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Function for loading parameters
def load_params(project_dir):
    with open(project_dir + "param.dat", mode="rb") as f:
        param_raw = f.read()
    with open(project_dir + "paramTypes.txt") as f:
        param_types = f.read().split('\n')[:-1]
    with open(project_dir + "paramNames.txt") as f:
        param_names = f.read().split('\n')[:-1]

    param_type_map = dict()
    param_type_map["Integer64"] = "q"
    param_type_map["Real64"] = "d"

    param_format_str = "".join(list(map(lambda t: param_type_map[t], param_types)))
    param_unpacked = struct.unpack_from(param_format_str, param_raw)
    param = {param_names[i] : param_unpacked[i] for i in range(len(param_unpacked))}

    return param


# Function for loading spectra
def load_list_of_arrays(project_dir, filename_template):
    template = re.compile(filename_template)
    filename_list = [filename for filename in os.listdir(project_dir) if template.match(filename)]
    pair_list = sorted([(int(template.match(filename).group(1)), filename) for filename in filename_list])
    array_list = [np.fromfile(project_dir + pair[1], dtype=np.float64) for pair in pair_list]

    return array_list


# Collection of methods for plotting spectra / showing slices
class PlottingUtils:
    def __init__(self, param):
        self.param = param
        self.k_IR = 2 * pi / self.param['L']
        self.k_points = self.k_IR * np.sqrt(np.arange(0, 3 * pow(self.param['N']//2, 2) + 1))

        # util for computing spectra
        self.multiplicity_list = np.zeros(3 * pow(param['N']//2, 2) + 1, dtype=np.int64)
        for i in range(-self.param['N']//2+1, self.param['N']//2 + 1):
            for j in range(-self.param['N']//2+1, self.param['N']//2 + 1):
                for k in range(-self.param['N']//2+1, self.param['N']//2 + 1):
                    self.multiplicity_list[i*i+j*j+k*k] += 1

        # constant log interval binning
        self.boundaries = np.unique(np.ceil(np.exp(2 * np.arange(0, log(self.param['N']/2), 0.06))).astype(np.int64))
        self.is_nonzero_bin = [self.multiplicity_list[self.boundaries[i]:self.boundaries[i+1]].sum() > 0 for i in range(0,len(self.boundaries)-1)] + [True]
        self.boundaries = self.boundaries[self.is_nonzero_bin]
        
        self.binned_multiplicities = np.array([self.multiplicity_list[self.boundaries[i]:self.boundaries[i+1]].sum() for i in range(0,len(self.boundaries)-1)])

        self.binned_k_points = self.k_points[self.boundaries][:-1]

    def compute_power_spectrum(self, spectrum):
        binned_power = np.array([spectrum[self.boundaries[i]:self.boundaries[i+1]].sum() for i in range(0,len(self.boundaries)-1)])
        return (
            self.binned_k_points,
            4 * pi * pow(self.binned_k_points * param['L'] / (2 * pi), 3) * binned_power / self.binned_multiplicities / pow(param['N'], 6)
        )

    def compute_power_spectrum_list(self, spectrum_list):
        return [self.compute_power_spectrum(spectrum) for spectrum in spectrum_list]
        



def compute_gradient(grid):
    delta_x = param["L"] / param["N"]
    
    dx_grid_head = ((grid[1,:,:] - grid[-1,:,:]) / (2 * delta_x)).reshape((1, param["N"], param["N"]))
    dx_grid_last = ((grid[-2,:,:] - grid[0,:,:]) / (2 * delta_x)).reshape((1, param["N"], param["N"]))
    dx_grid_mid = (grid[2:,:,:] - grid[:(-2),:,:]) / (2 * delta_x)
    dx_grid = np.concatenate((dx_grid_head, dx_grid_mid, dx_grid_last), axis=0)

    dy_grid_head = ((grid[:,1,:] - grid[:,-1,:]) / (2 * delta_x)).reshape((param["N"], 1, param["N"]))
    dy_grid_last = ((grid[:,-2,:] - grid[:,0,:]) / (2 * delta_x)).reshape((param["N"], 1, param["N"]))
    dy_grid_mid = (grid[:,2:,:] - grid[:,:(-2),:]) / (2 * delta_x)
    dy_grid = np.concatenate((dy_grid_head, dy_grid_mid, dy_grid_last), axis=1)

    dz_grid_head = ((grid[:,:,1] - grid[:,:,-1]) / (2 * delta_x)).reshape((param["N"], param["N"], 1))
    dz_grid_last = ((grid[:,:,-2] - grid[:,:,0]) / (2 * delta_x)).reshape((param["N"], param["N"], 1))
    dz_grid_mid = (grid[:,:,2:] - grid[:,:,:(-2)]) / (2 * delta_x)
    dz_grid = np.concatenate((dz_grid_head, dz_grid_mid, dz_grid_last), axis=2)
    
    return [dx_grid, dy_grid, dz_grid]

    

def filter_2d(array2d, max_s):
    q_fft_2 = fft.rfft2(array2d)
    for a in range(param["N"]):
        for b in range(param["N"]//2+1):
            a_shifted = a if (a <= param["N"]//2) else (param["N"]-a)
            b_shifted = b if (b <= param["N"]//2) else (param["N"]-b)
            s = a_shifted**2 + b_shifted**2
            if s > max_s:
                q_fft_2[a, b] = 0
    q_filtered_2 = fft.irfft2(q_fft_2)
    return q_filtered_2


def filter_3d(f, max_s):
    f_fft = fft.rfftn(f)
    for a in range(param["N"]):
        for b in range(param["N"]):
            for c in range(param["N"]//2+1):
                a_shifted = a if (a <= param["N"]//2) else (param["N"]-a)
                b_shifted = b if (b <= param["N"]//2) else (param["N"]-b)
                c_shifted = c if (c <= param["N"]//2) else (param["N"]-c)
                s = a_shifted**2 + b_shifted**2 + c_shifted**2
                if s > max_s:
                    f_fft[a, b, c] = 0
    return fft.irfftn(f_fft)


def compute_negative_gradient_averaged(tau_filename:str):
    tau = np.fromfile(tau_filename, dtype=np.float64).reshape((param["N"], param["N"], param["N"]))
    gradients = compute_gradient(tau)
    return [-f.mean(axis=0) for f in gradients]


def compute_delta_averaged(rho_filename:str):
    rho = np.fromfile(rho_filename, dtype=np.float64).reshape((param["N"], param["N"], param["N"]))
    rho_mean = rho.mean()
    delta = (rho / rho_mean) - 1.0
    return delta.mean(axis=0)


def compute_momentum_averaged(varphi_filename:str, dt_varphi_filename:str):
    varphi = np.fromfile(varphi_filename, dtype=np.float64).reshape((param["N"], param["N"], param["N"]))
    dt_varphi = np.fromfile(dt_varphi_filename, dtype=np.float64).reshape((param["N"], param["N"], param["N"]))
    dxyz_varphi = compute_gradient(varphi)
    q_xyz = [-di_varphi * dt_varphi for di_varphi in dxyz_varphi]
    return [f.mean(axis=0) for f in q_xyz]

# Load data
project_dir = "output/scalar_IC/"
param = load_params(project_dir)
utils = PlottingUtils(param)

# varphi = np.fromfile(project_dir + r"varphi.dat", dtype=np.float64).reshape((param["N"], param["N"], param["N"]))
# dt_varphi = np.fromfile(project_dir + r"dt_varphi.dat", dtype=np.float64).reshape((param["N"], param["N"], param["N"]))


# dxyz_varphi = compute_gradient(varphi)
# dyz_varphi_averaged = [f.mean(axis=0) for f in dxyz_varphi]

# q_xyz = [-di_varphi * dt_varphi for di_varphi in dxyz_varphi]
# q_averaged = [f.mean(axis=0) for f in q_xyz]


# rho = np.fromfile(project_dir + r"rho.dat", dtype=np.float64).reshape((param["N"], param["N"], param["N"]))
# rho_mean = rho.mean()
# rho_averaged = rho.mean(axis=0)
# delta = (rho / rho_mean) - 1.0
# delta_averaged = delta.mean(axis=0)


v_averaged = compute_negative_gradient_averaged(project_dir + r"tau.dat")

delta_averaged = compute_delta_averaged(project_dir + r"rho.dat")
q_averaged = compute_momentum_averaged(project_dir + r"varphi.dat", project_dir + r"dt_varphi.dat")

delta_averaged_old = compute_delta_averaged(project_dir + r"rho_old.dat")
q_averaged_old = compute_momentum_averaged(project_dir + r"varphi_old.dat", project_dir + r"dt_varphi_old.dat")

import gc
gc.collect()

#q_averaged_filtered = [filter_k(q, 5) for q in q_averaged]
#q_filtered = [filter_3d(q, 10) for q in q_xyz]

# Font Settings
font_path = font_manager.findfont("Latin Modern Roman")
font = matplotlib.font_manager.FontProperties(fname=font_path)
plt.rcParams.update({
    "text.usetex": True
})


# Color scheme for visualizing slices
xList = [0., 0.166667, 0.333333, 0.499999, 0.5, 0.500001, 0.666667, 0.833333, 1.]
rgbList = [
    [0.260487, 0.230198, 0.392401, 0.964837, 1, 0.95735, 0.913252,  0.860243, 1.],
    [0.356, 0.499962, 0.658762, 0.982332, 1, 0.957281, 0.790646, 0.558831, 0.42],
    [0.891569, 0.848188, 0.797589, 0.98988, 1, 0.896269, 0.462837, 0.00695811, 0.]
]
cmbDict = {
    'red': [(xList[i], rgbList[0][i], rgbList[0][i]) for i in range(len(xList))],
    'green': [(xList[i], rgbList[1][i], rgbList[1][i]) for i in range(len(xList))],
    'blue': [(xList[i], rgbList[2][i], rgbList[2][i]) for i in range(len(xList))]
}
cmbColor = matplotlib.colors.LinearSegmentedColormap("cmb", cmbDict)
colorNorm = matplotlib.colors.TwoSlopeNorm(0, vmin=-0.5, vmax=1.0)

whiteColor = matplotlib.colors.ListedColormap([(1,1,1)])



# Plotting
slice_ticks = np.array([0, 5, 10, 15])
slice_labels = list(map(lambda x: '$' + str(x) + '$', slice_ticks))




# Function to plot one snapshot
def plot_slice(ax, grid, time=None, show_colorbar=True):
    cax = ax.imshow(grid, cmap=cmbColor, norm=colorNorm, aspect='equal', origin='lower', extent=(0,param['L'],0,param['L']))
    ax.tick_params(axis="both",which="both",bottom=True,top=False,left=False,right=False,labelbottom=True,labeltop=False,labelleft=False,labelright=False,direction='in',length=2.0,width=0.5,reset=True)
    ax.set_xticks(slice_ticks) # / (param['L'] / param['N']))
    ax.set_xticklabels(slice_labels)
    for label in ax.get_xticklabels():
        label.set_fontproperties(font)
    ax.set_xlabel(r'$mx$',fontsize=15)
    if show_colorbar:
        cax_colorbar = make_axes_locatable(ax).append_axes("right", size="5%", pad=0)
        cbar = ax.figure.colorbar(cax, cax=cax_colorbar, ax=ax)
    


# matplotlib.rcParams['axes.linewidth'] = 0.5

spacing = 8
quiver_scale = 2

# Plot a row
fig = plt.figure(figsize=(6.1,2))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0, hspace=0)


# Plot 1: velocity field
ax = fig.add_subplot(gs[0, 0])
x = np.linspace(0, param["L"] - param["L"] / param["N"], param["N"])
X, Y = np.meshgrid(x, x, indexing='ij')
# u = np.zeros(q_averaged[1].shape)
# v = -np.sin(2 * np.pi * Y / param["L"])
u = v_averaged[1]
v = v_averaged[2]


X = X[(spacing//2)::spacing, (spacing//2)::spacing]
Y = Y[(spacing//2)::spacing, (spacing//2)::spacing]
u = u[(spacing//2)::spacing, (spacing//2)::spacing]
v = v[(spacing//2)::spacing, (spacing//2)::spacing]
qv2 = ax.quiver(X, Y, u, v, angles='xy', pivot='mid', scale=1e-1, scale_units='dots')
#plt.quiverkey(qv2, -0.1, 0.9, 1, r'$\vec{v}$', coordinates='axes', labelpos='N')
ax.tick_params(axis="both",which="both",bottom=True,top=False,left=True,right=False,labelbottom=True,labeltop=False,labelleft=True,labelright=False,direction='in',length=2.0,width=0.5,reset=True)
ax.set_xlim(0, param["L"])
ax.set_ylim(0, param["L"])
ax.set_xlabel(r'$mx$',fontsize=15)
ax.set_ylabel(r'$my$',fontsize=15)
ax.set_xticks(slice_ticks)
ax.set_yticks(slice_ticks)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
#font_properties = {'family': 'Times New Roman', 'size': 15}
ax.set_title(r'velocity field', fontsize=15)
             #fontdict=font_properties)
 #            $fontsize=15, fontproperties=font)
 #fontfamily='latin')


# Plot 2: before boost
ax = fig.add_subplot(gs[0, 1])


x = np.linspace(0, param["L"] - param["L"] / param["N"], param["N"])
X, Y = np.meshgrid(x, x, indexing='ij')
u = q_averaged_old[1]
v = q_averaged_old[2]

X = X[(spacing//2)::spacing, (spacing//2)::spacing]
Y = Y[(spacing//2)::spacing, (spacing//2)::spacing]
u = u[(spacing//2)::spacing, (spacing//2)::spacing]
v = v[(spacing//2)::spacing, (spacing//2)::spacing]

#ax.quiver(X, Y, u, v, angles='xy', pivot='mid')
ax.quiver(X, Y, u, v, angles='xy', pivot='mid', scale=quiver_scale, scale_units='dots')

to_show = delta_averaged_old
plot_slice(ax, to_show.transpose(), show_colorbar=False)

ax.set_xlim(0, param["L"])
ax.set_ylim(0, param["L"])
ax.set_title(r'before boost', fontsize=15)


# Plot 3: after boost
ax = fig.add_subplot(gs[0, 2])

x = np.linspace(0, param["L"] - param["L"] / param["N"], param["N"])
X, Y = np.meshgrid(x, x, indexing='ij')
u = q_averaged[1]
v = q_averaged[2]
# u = q_xyz[1][0]
# v = q_xyz[2][0]
# u = q_averaged_filtered[1]
# v = q_averaged_filtered[2]

X = X[(spacing//2)::spacing, (spacing//2)::spacing]
Y = Y[(spacing//2)::spacing, (spacing//2)::spacing]
u = u[(spacing//2)::spacing, (spacing//2)::spacing]
v = v[(spacing//2)::spacing, (spacing//2)::spacing]

qv = ax.quiver(X, Y, u, v, angles='xy', pivot='mid', scale=quiver_scale, scale_units='dots')
ax.quiverkey(qv, 1.0, -0.1, 1e1, r'$|\vec{q}| / m^4 = 10$', coordinates='axes', labelpos='S', labelsep=0.05, fontproperties={'size':5})

#to_show = filter_k(delta_averaged, 3)
to_show = delta_averaged
plot_slice(ax, to_show.transpose())

ax.set_xlim(0, param["L"])
ax.set_ylim(0, param["L"])
ax.set_title(r'after boost', fontsize=15)


# Assemble
plt.savefig('temp_figure.pdf', bbox_inches='tight', dpi=500)
plt.clf()






x_bounds = [1.0e-1, 60]
y_bounds = [1e-4, 1e0]
log_aspect_ratio = log(y_bounds[1]/y_bounds[0]) / log(x_bounds[1]/x_bounds[0])

mt_text_pos = [1.5e-2, 2]


# Function to plot one spectrum
def plot_spectrum(ax, power_spectrum, initial_power_spectrum=None, time=None):
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    if initial_power_spectrum:
        ax.loglog(*initial_power_spectrum, linewidth=1.5, color='0.8')
    ax.loglog(*power_spectrum, linewidth=1.5, color='tab:orange')
    ax.set_aspect(0.5 / log_aspect_ratio)
    ax.tick_params(axis="both",which="both",bottom=True,top=True,left=True,right=False,direction='in',length=5.0,width=0.5,reset=True)
    
    if param['H1'] != 0:
        ax.set_xlabel(r'$k/a_i m$',fontsize=15)
    else:
        ax.set_xlabel(r'$k/m$',fontsize=15)
        
    ax.set_ylabel(r'$\Delta_\delta^2(t,k)$',fontsize=15)
    
    # ax.text(*mt_text_pos,r'$mt={:.0f}$'.format(param['m'] * time),fontsize=10,color='0')


fig = plt.figure(figsize=(6.1,2))
gs = fig.add_gridspec(1, 1, width_ratios=[1], wspace=0, hspace=0)

ax = fig.add_subplot(gs[0, 0])

rho_spectrum_old = np.fromfile(project_dir + "rho_spectrum_old.dat", dtype=np.float64)
delta_spectrum_old = rho_spectrum_old / (rho_spectrum_old[0] / pow(param['N'], 6))
power_spectrum_old = utils.compute_power_spectrum(delta_spectrum_old)

rho_spectrum = np.fromfile(project_dir + "rho_spectrum.dat", dtype=np.float64)
delta_spectrum = rho_spectrum / (rho_spectrum[0] / pow(param['N'], 6))
power_spectrum = utils.compute_power_spectrum(delta_spectrum)

# ax.loglog(*power_spectrum, linewidth=1.5, color='tab:orange')
# ax.set_xlim(*x_bounds)
# ax.set_ylim(*y_bounds)

plot_spectrum(ax, power_spectrum, initial_power_spectrum=power_spectrum_old)

plt.savefig('spectrum_temp_figure.pdf', bbox_inches='tight', dpi=500)
plt.clf()


varphi_spectrum = np.fromfile(project_dir + "varphi_spectrum.dat", dtype=np.float64)
varphi_power_spectrum = utils.compute_power_spectrum(varphi_spectrum)
varphi_spectrum_old = np.fromfile(project_dir + "varphi_spectrum_old.dat", dtype=np.float64)
varphi_power_spectrum_old = utils.compute_power_spectrum(varphi_spectrum_old)

fig = plt.figure(figsize=(6.1,2))
gs = fig.add_gridspec(1, 1, width_ratios=[1], wspace=0, hspace=0)
ax = fig.add_subplot(gs[0, 0])
# ax.set_xlim(*x_bounds)
plot_spectrum(ax, varphi_power_spectrum, initial_power_spectrum=varphi_power_spectrum_old)
ax.set_ylim([1e-3,1e1])
plt.savefig('spectrum_temp_figure.pdf', bbox_inches='tight', dpi=500)
plt.clf()
