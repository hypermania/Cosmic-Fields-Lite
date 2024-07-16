import numpy as np
from numpy import sqrt, pi, sin, cos, log, log10, exp, tanh, sinh, cosh
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
        



# Load data
#project_dir = "output/FS_Without_Gravity/"
#project_dir = "output/Growth_and_FS_2_small_time_step/"
project_dir = "output/Soliton_1/"
param = load_params(project_dir)

rho_spectrum_list = load_list_of_arrays(project_dir, "rho_spectrum_([0-9]+).dat")
varphi_plus_spectrum_list = load_list_of_arrays(project_dir, "varphi_plus_spectrum_([0-9]+).dat")
delta_spectrum_list = [spectrum / (spectrum[0] / pow(param['N'], 6)) for spectrum in rho_spectrum_list]

rho_average_list = load_list_of_arrays(project_dir, "rho_axis_average_([0-9]+).dat")
delta_average_grid_list = [np.reshape(rho / rho.mean() - 1.0, [param['N'],param['N']]) for rho in rho_average_list]

t_list = np.fromfile(project_dir + "t_list.dat", dtype=np.float64)
#t_list = np.linspace(0, 12500, 12500 // 50 + 1)


wkb_rho_spectrum_list = load_list_of_arrays(project_dir, "wkb_rho_spectrum_([0-9]+).dat")
wkb_varphi_plus_spectrum_list = load_list_of_arrays(project_dir, "wkb_varphi_plus_spectrum_([0-9]+).dat")
wkb_delta_spectrum_list = [spectrum / (spectrum[0] / pow(param['N'], 6)) for spectrum in wkb_rho_spectrum_list]

wkb_rho_average_list = load_list_of_arrays(project_dir, "wkb_rho_axis_average_([0-9]+).dat")
wkb_delta_average_grid_list = [np.reshape(rho / rho.mean() - 1.0, [param['N'],param['N']]) for rho in wkb_rho_average_list]

wkb_t_list = np.fromfile(project_dir + "wkb_t_list.dat", dtype=np.float64)


# Combine simulated data and WKB data
combined_delta_spectrum_list = np.concatenate((delta_spectrum_list, wkb_delta_spectrum_list))
combined_varphi_plus_spectrum_list = np.concatenate((varphi_plus_spectrum_list, wkb_varphi_plus_spectrum_list))
combined_delta_average_grid_list = np.concatenate((delta_average_grid_list, wkb_delta_average_grid_list))
combined_t_list = np.concatenate((t_list, wkb_t_list))


utils = PlottingUtils(param)
#utils.multiplicity_list.tofile("temp_multiplicity_list.dat")

delta_power_spectrum_list = utils.compute_power_spectrum_list(delta_spectrum_list)
#combined_delta_power_spectrum_list = utils.compute_power_spectrum_list(combined_delta_spectrum_list)


del rho_average_list
del rho_spectrum_list
del varphi_plus_spectrum_list
del delta_spectrum_list
del wkb_rho_spectrum_list
del wkb_varphi_plus_spectrum_list
del wkb_delta_spectrum_list


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
    
    ax.text(*mt_text_pos,r'$mt={:.0f}$'.format(param['m'] * time),fontsize=10,color='0')


def plot_slice(ax, grid, time=None):
    cax = ax.imshow(grid, cmap=cmbColor, norm=colorNorm, aspect='equal')
    ax.tick_params(axis="both",which="both",bottom=True,top=False,left=False,right=False,labelbottom=True,labeltop=False,labelleft=False,labelright=False,direction='in',length=5.0,width=0.5,reset=True)
    ax.set_xticks(slice_ticks / (param['L'] / param['N']))
    ax.set_xticklabels(slice_labels)
    for label in ax.get_xticklabels():
        label.set_fontproperties(font)
    if param['H1'] != 0:
        ax.set_xlabel(r'$a_i mx$',fontsize=15)
    else:
        ax.set_xlabel(r'$mx$',fontsize=15)
    cax_colorbar = make_axes_locatable(ax).append_axes("right", size="5%", pad=0)
    cbar = ax.figure.colorbar(cax, cax=cax_colorbar, ax=ax)
    #cbar.set_label(r'$\delta$')
    



# Plotting
x_bounds = [1.0e-2, 4]
y_bounds = [1e-4, 1e1]
log_aspect_ratio = log(y_bounds[1]/y_bounds[0]) / log(x_bounds[1]/x_bounds[0])

mt_text_pos = [1.5e-2, 2]

slice_ticks = np.array([0, 100, 200, 300])
slice_labels = list(map(lambda x: '$' + str(x) + '$', slice_ticks))


"""
# Without gravity
fig = plt.figure(figsize=(6.4,7.2))
gs = fig.add_gridspec(3, 2, width_ratios=[2, 1.05], wspace=0, hspace=0)

initial_power_spectrum = combined_delta_power_spectrum_list[0]

# 1st row
ax = fig.add_subplot(gs[0, 0])
plot_spectrum(ax, combined_delta_power_spectrum_list[0], time=combined_t_list[0])

ax = fig.add_subplot(gs[0, 1])
plot_slice(ax, combined_delta_average_grid_list[0])


# 2nd row
ax = fig.add_subplot(gs[1, 0])
plot_spectrum(ax, np.array(combined_delta_power_spectrum_list[720])[:,:68], initial_power_spectrum=initial_power_spectrum, time=combined_t_list[720])

ax = fig.add_subplot(gs[1, 1])
plot_slice(ax, combined_delta_average_grid_list[720])


# 3rd row
ax = fig.add_subplot(gs[2, 0])
plot_spectrum(ax, np.array(combined_delta_power_spectrum_list[-1])[:,:68], initial_power_spectrum=initial_power_spectrum, time=combined_t_list[-1])

ax = fig.add_subplot(gs[2, 1])
plot_slice(ax, combined_delta_average_grid_list[-1])

plt.savefig('../../Plots/delta_spectrum_without_gravity.pdf', bbox_inches='tight', dpi=500)
#plt.savefig('../../Plots/fig_python_test.pdf', bbox_inches='tight', dpi=500)
#plt.show()
"""


"""
# With gravity
#fig = plt.figure(figsize=(6,9))
#fig = plt.figure(figsize=(6.4,9.6))
fig = plt.figure(figsize=(6.4,9.6))
gs = fig.add_gridspec(4, 2, width_ratios=[2, 1.05], wspace=0, hspace=0)

initial_power_spectrum = combined_delta_power_spectrum_list[0]

# 1st row
ax = fig.add_subplot(gs[0, 0])
plot_spectrum(ax, combined_delta_power_spectrum_list[0], time=combined_t_list[0])

ax = fig.add_subplot(gs[0, 1])
plot_slice(ax, combined_delta_average_grid_list[0])


# 2nd row
ax = fig.add_subplot(gs[1, 0])
plot_spectrum(ax, np.array(combined_delta_power_spectrum_list[14])[:,:68], initial_power_spectrum=initial_power_spectrum, time=combined_t_list[14])

ax = fig.add_subplot(gs[1, 1])
plot_slice(ax, combined_delta_average_grid_list[14])


# 3rd row
ax = fig.add_subplot(gs[2, 0])
#plot_spectrum(ax, combined_delta_power_spectrum_list[720], initial_power_spectrum=initial_power_spectrum, time=combined_t_list[720])
plot_spectrum(ax, np.array(combined_delta_power_spectrum_list[720])[:,:68], initial_power_spectrum=initial_power_spectrum, time=combined_t_list[720])

ax = fig.add_subplot(gs[2, 1])
plot_slice(ax, combined_delta_average_grid_list[720])

# 4th row
ax = fig.add_subplot(gs[3, 0])
plot_spectrum(ax, np.array(combined_delta_power_spectrum_list[-1])[:,:68], initial_power_spectrum=initial_power_spectrum, time=combined_t_list[-1])

ax = fig.add_subplot(gs[3, 1])
plot_slice(ax, combined_delta_average_grid_list[-1])

plt.savefig('../../Plots/delta_spectrum_growth_and_fs_2.pdf', bbox_inches='tight', dpi=500)

"""




#Soliton
x_bounds = [5e-3, 2]
y_bounds = [1e-4, 1e3]
log_aspect_ratio = log(y_bounds[1]/y_bounds[0]) / log(x_bounds[1]/x_bounds[0])

mt_text_pos = [7e-3, 1e2]


slice_ticks = np.array([0, 250, 500, 750])
slice_labels = list(map(lambda x: '$' + str(x) + '$', slice_ticks))
#slice_ticks = np.array([0, 25, 50, 75])


fig = plt.figure(figsize=(6.4,9.6))
gs = fig.add_gridspec(4, 2, width_ratios=[2, 1.05], wspace=0, hspace=0)

initial_power_spectrum = delta_power_spectrum_list[0]

# 1st row
ax = fig.add_subplot(gs[0, 0])
plot_spectrum(ax, delta_power_spectrum_list[0], time=t_list[0])

ax = fig.add_subplot(gs[0, 1])
plot_slice(ax, delta_average_grid_list[0])


# 2nd row
idx = 20
ax = fig.add_subplot(gs[1, 0])
plot_spectrum(ax, np.array(delta_power_spectrum_list[idx]), initial_power_spectrum=initial_power_spectrum, time=t_list[idx])
#ax.plot([1 / (param['k_ast'] / param['m'] * t_list[idx])] * 2, [1e-4,1e4], color='k')

ax = fig.add_subplot(gs[1, 1])
plot_slice(ax, delta_average_grid_list[idx])


# 3rd row
idx = 60
ax = fig.add_subplot(gs[2, 0])
plot_spectrum(ax, np.array(delta_power_spectrum_list[idx]), initial_power_spectrum=initial_power_spectrum, time=t_list[idx])
#ax.plot([1 / (param['k_ast'] / param['m'] * t_list[idx])] * 2, [1e-4,1e4], color='k')

ax = fig.add_subplot(gs[2, 1])
plot_slice(ax, delta_average_grid_list[idx])


# 4th row
idx = -1
ax = fig.add_subplot(gs[3, 0])
plot_spectrum(ax, np.array(delta_power_spectrum_list[idx]), initial_power_spectrum=initial_power_spectrum, time=t_list[idx])
#ax.plot([1 / (param['k_ast'] / param['m'] * t_list[idx])] * 2, [1e-4,1e4], color='k')

ax = fig.add_subplot(gs[3, 1])
plot_slice(ax, delta_average_grid_list[idx])


# Save
plt.savefig('../../Plots/delta_spectrum_soliton.pdf', bbox_inches='tight', dpi=500)
#plt.savefig('../../Plots/fig_python_test.pdf', bbox_inches='tight', dpi=500)
#plt.show()
plt.close()


