import os
import numpy as np
import string
from scipy.optimize import curve_fit
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import matplotlib.pyplot as plt
from draw import Draw


class Txt_Reading:
    def __init__(self, folder, f_MAX) -> None:
        self.KBT = 4.11     # pN*nm
        self.d = 2          # nm
        self.P = 1.35       # ± 0.05 nm -> persistence length
        self.d_aa = 0.58    # ± 0.02 nm/base -> distance between consecutive nucleotides
        self.loading_rate = 6   # pN/s -> useful to discard useless data. This value can be changed to 5 or 4.5
        self.folder = folder
        # self.dir_name = 'G:/Il mio Drive/Fisica/Dispense Università/2 anno/Zaltron & Xavi/' + dir_name
        self.dir_name = 'E:/Zaltron & Xavi/' + self.folder + '/'
        self.f_MAX = str(f_MAX)
        self.name = f'pull{self.f_MAX}'
        # self.dictionary = {'10':9, '15':9, '20':9, '25':9, '30':9, '35':9}
        # self.fit = {1:15, 2:15, 3:15, 4:10, 5:10, 6:7, 7:7, 8:5, 9:5, 10:10}
        # self.decision = 3.0 if int(self.f_MAX) < 15 else -0.2

        self.working_dir = f'Data/{self.folder}/{self.f_MAX}'
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            self._createTxt()

        self.draw = Draw(self.folder, self.f_MAX)


    def readTxt(self, number = 1, N = 1, ty = 'u', print_out = True, forced_reshaped = 0, graph = False, initial_t_time = False):
        self.number = number    # molecule number
        self.ty = ty    # file type: unfolding (u) or folding (f) 
        self.path = f'{self.working_dir}/{self.number}/{N}_{self.ty}.txt'

        self.file = np.loadtxt(self.path)

        # take account only of rows with f = 100 kHz
        self.file = self.file[:-1][np.diff(self.file[:, 0]) == 1, :]

        self.λ_original = self.file[:, -1] if ty == 'u' else self.file[:, -1][::-1]
        self.force_Y_original = np.abs(self.file[:, 2]) if ty == 'u' else np.abs(self.file[:, 2][::-1])
        self.time = self.file[:, -3] if ty == 'u' else self.file[:, -3][::-1]
        self.t_initial = self._set_initial_time() if initial_t_time else (self.time[0] if ty == 'u' else self.time[-1])
        self.time -= self.t_initial # different initial time for each file, unique for each molecule

        # Correction just to preserve the next part of the analysis if for any chance
        # the x-axis is shifted back, into negatives numbers
        # Normalization
        self.λ = (self.λ_original - np.min(self.λ_original))/(np.max(self.λ_original)-np.min(self.λ_original))

        # ???
        self.fitting_points = 15  if int(self.f_MAX) < 15 else 10 # number of points used for linear fits
        self.N_fits = 0     # Number of fits performed: it can be 0 (no results), 1 (the molecule doesn't open/close), 2 (standard case)
        ### ???
        self.best, self.linear = 0, 0

        # I want to work in a specific range of forces: discard all values f_Y > 10 pN
        try:
            self.filter_10 = np.where(self.force_Y_original >= 10.)[0][0]
            self.force_Y = self.force_Y_original[:self.filter_10]
            self.λ = self.λ[:self.filter_10]
        except:
            self._errors()

        # ???
        # if not forced_reshaped:
        #     # default reshape of 5
        #     self.force_Y, self.λ, self.time = self.reshape(n_point5)

        # try:
        self.analysis(N=N, print_out=print_out, forced_reshaped=forced_reshaped, graph=graph)
        # except:
        #     self._errors()


        columns = ["f_rupture", "f_rupture_next", "x_ssDNA", "N_nucleotides", "t_0", "λ_0", "k_U", "k_F", "N_fits"]
        self.params_df = pd.DataFrame([self.params], columns=columns)

        return self.file
    

    def analysis(self, N, print_out=True, forced_reshaped=False, graph=False):
        conditions = [np.abs(self.time[-1] - self.time[0]) < (np.max(self.force_Y_original) - np.min(self.force_Y_original))/self.loading_rate, 
                 max(self.force_Y_original)+1 < int(self.f_MAX), 
                 min(self.force_Y_original) > int(self.f_MAX)/3, 
                 self.λ.size < 50, 
                 self.force_Y_original[-50:-1].mean() < int(self.f_MAX)/2,
                 max(self.force_Y_original) > int(self.f_MAX)+4,
                 self.λ[np.where(self.force_Y_original == min(self.force_Y_original))[0][0]]>0.1]
        if  sum(conditions) > 0:
            self._errors()
            # Noise or other useless data
            if graph:
                self.make_plot(save_fig=False, folder=self.folder, number=self.number, N=self.path.split('_')[0].split('/')[-1], txt=False)
            return -1
        
        # In order to find the rip, I perform a reshape of the data, here set to 5
        # This to clear data from noise and better identify the jump
        # It's something optional that can be removed or amplified
        # ???
        # force_Y_reshaped, λ_reshaped, time_reshaped = self.reshape(n_points=5)
        
        try:
            λ_0, index, f_rupture = self.find_λ_0(self.number, N)
        except:
            # No better result found
            # Could be a linear fit
            output = self.draw.identify_by_hand(self.λ, self.force_Y)

        if output == 1:
            # linear fit
            self.N_fits = 1
            self._errors()
            self.theor_f = self._linear_fit()

        else:
            # Explore the highlighted area
            new_λ = output[:, 0]
            new_force_Y = output[:, 1]
            print('#ToDo: play with the selected area')
            # double-jump
            self.N_fits = 2
            self.params = self.f_rupture + self.f_rupture_next + self.x_ssDNA + self.N_nucleotides + self.t_0 + self.λ_0 + self.popt_pre.tolist() + self.popt_post.tolist() + [self.N_fits]
            self.theor_f = self._heviside_fitting(self.λ, *self.params[5:-1])   
                


        if graph:
            ind = int(self.index[0])
            plt.plot(self.λ, self.force_Y, label = 'Data')
            plt.plot(self.λ[ind-self.fitting_points:ind+self.fitting_points], self.theor_f[ind-self.fitting_points:ind+self.fitting_points], lw=2.5, label = 'Fit')
            plt.xlabel('$\lambda \\: [nm]$')
            plt.ylabel('$f_Y \\: [pN]$')
            plt.grid()
            plt.legend()
            plt.title(f'{self.path}')
            plt.show()  


        

    
    def find_λ_0(self, number, N, mask=[], plot_diff=True):
        # First we "randomly" select the rip point, by selecting the best λ_0 
        # within a range of values such that the χ² is the minimum possible
        filt = [np.where(self.force_Y>=0)[0].tolist()[5:] and np.where(self.force_Y<=8)[0].tolist()][0] if len(mask) == 0 else mask
        # tot_length = self.λ.size
        bootstrap_indexes = np.linspace(filt[0], filt[-1], 50, dtype=int)
        chi = np.Inf
        best = None
        self.index = [5]
        for index in bootstrap_indexes:
            eval = self.chi_squared(index)
            if eval < chi:
                chi = eval 
                best = index
            else:
                pass

        # Now that I have a possible/reasonable λ_0, I want to see if there is something better:
        fitting_points = self.fitting_points if best > self.fitting_points else best
        λ_0 = self.λ[index]
        f_rupture, f_rupture_next = self.force_Y[best:best+2]
        popt_pre, popt_post = self._compute_fit(best, fitting_points, self.λ, self.force_Y)
        self.t_0, k_eff, self.x_ssDNA, self.N_nucleotides = self._compute_interesting_variables(self.time, popt_pre, popt_post, best, [f_rupture], [f_rupture_next])
        params = [self.λ[best]] + popt_pre.tolist() + popt_post.tolist()  
        theor = self._heviside_fitting(self.λ, *params)
        diff = self.force_Y - theor
        # For a 'u' curve I expect that the experimental point is higher than the fitting line
        # otherwise lower for an 'f' curve
        std = 3*np.std(diff)
        theor_std = np.sqrt(self.KBT*k_eff[0])

        plt.plot(self.λ, self.force_Y)
        plt.plot(self.λ, theor)
        plt.show()

        if plot_diff:
            self.draw.draw_fit_diff(self.λ, diff, std, theor_std, [λ_0, f_rupture], number, N, save=False)
        

        # find new rip
        update = 0
        if self.ty == 'u':
            wh = (np.where(diff >= std)[0].tolist() and np.where(diff >= theor_std)[0].tolist())[0]  
            if not wh:
                wh = np.where(diff >= std)[0].tolist()
                if not wh:
                    # No better point found: stick with the previous results
                    # return ...
                    self.linear += 1
                    return -1
                else:
                    self.index = [wh]
                    update = 1
            else:
                self.index = [wh]
                update = 1
        else:
            wh = (np.where(diff <= -std)[0].tolist() and np.where(diff <= -theor_std)[0].tolist())[0]  
            if not wh:
                wh = np.where(diff <= -std)[0].tolist()
                if not wh:
                    # No better point found: stick with the previous results
                    # return ...
                    self.linear += 1
                    return -1
                else:
                    self.index = [wh]
                    update = 1
            else:
                self.index = [wh]
                update = 1
        print(f'wh = {wh}')
        index = self.index[0]
        print(f'index = {index}')
        λ_0 = self.λ[index]
        f_rupture = self.force_Y[index] 
        fitting_points = self.fitting_points if index > self.fitting_points else index
        f_rupture_next = (popt_post[0]*self.λ[index:index+fitting_points] + popt_post[1])[0]
        
        if update:
            fitting_points = self.fitting_points if best > self.fitting_points else best
            popt_pre, popt_post = self._compute_fit(best, fitting_points, self.λ, self.force_Y)
            self.t_0, k_eff, self.x_ssDNA, self.N_nucleotides = self._compute_interesting_variables(self.time, popt_pre, popt_post, index, [f_rupture], [f_rupture_next])

        self.f_rupture = [f_rupture]
        self.f_rupture_next = [f_rupture_next]
        self.fitting_points = fitting_points
        self.popt_pre, self.popt_post = popt_pre, popt_post
        self.k_F, self.k_U = k_eff if self.ty == 'u' else k_eff[::-1]


        return [λ_0], np.array([best]), [f_rupture]

        


    def _compute_interesting_variables(self, time, popt_pre, popt_post, index, f_rupture, f_rupture_next):
        # This function computes important variables such as:
        # t_0: when the jump happens
        # k_eff: effective k
        # x_ssDNA: length of the single strand DNA
        # N_nucleotides: Number of nucleoteotides
        t_0 = [time[index]]
        if self.ty == 'u':
            k_eff_F = popt_pre[0]/(np.max(self.λ_original)-np.min(self.λ_original)) 
            k_eff_U = popt_post[0]/(np.max(self.λ_original)-np.min(self.λ_original))
            x_ssDNA = np.array([np.abs(f_rupture[0]-f_rupture_next[0])/k_eff_F + self._calculation_x_d(f_rupture[0])]).tolist()
            N_nucleotides = np.array([x_ssDNA[0] / (self.x_WLC_f(f_rupture[0]) * self.d_aa)]).tolist()
            return t_0, [k_eff_F, k_eff_U], x_ssDNA, N_nucleotides
        else:
            k_eff_U = popt_pre[0]/(np.max(self.λ_original)-np.min(self.λ_original)) 
            k_eff_F = popt_post[0]/(np.max(self.λ_original)-np.min(self.λ_original))
            x_ssDNA = np.array([np.abs(f_rupture[0]-f_rupture_next[0])/k_eff_U + self._calculation_x_d(f_rupture[0])]).tolist()
            N_nucleotides = np.array([x_ssDNA[0] / (self.x_WLC_f(f_rupture[0]) * self.d_aa)]).tolist()
        return t_0, [k_eff_U, k_eff_F], x_ssDNA, N_nucleotides
        

    def _heviside_fitting(self, x, λ_0, a, b, c, d):
        return (x*a + b) * np.heaviside(λ_0 - x, 0.5) + (c*x + d) * np.heaviside(x - λ_0, 0.5) 
    
    def _compute_fit(self, ind, fitting_points, λ, force_Y):
        pre_λ = λ[ind-fitting_points:ind]
        pre_f = force_Y[ind-fitting_points:ind]
        popt_pre, _ = curve_fit(lambda x, a, b: x*a+b, pre_λ, pre_f)
        post_λ = λ[ind:ind+fitting_points]
        post_f = force_Y[ind:ind+fitting_points]
        popt_post, _ = curve_fit(lambda x, a, b: x*a+b, post_λ, post_f)
        return popt_pre, popt_post

    def chi_squared(self, index=0, mode=''):
        ind = self.index[0] if not index else index
        fitting_points = self.fitting_points if ind > self.fitting_points else ind
        std = np.std(self.force_Y[ind-fitting_points:ind+fitting_points])
        if mode:
            # linear
            expected = self._linear_fit(ran=[ind-fitting_points, ind+fitting_points])
            return (np.sum(((self.force_Y[ind-fitting_points:ind+fitting_points] - expected)**2)/std**2))
        # default: double-jump
        [a, b], [c, d] = self._compute_fit(ind=ind, fitting_points=fitting_points, λ=self.λ, force_Y=self.force_Y)
        expected = self._heviside_fitting(self.λ[ind-fitting_points:ind+fitting_points], self.λ[ind], a, b, c, d)
        return np.sum(((self.force_Y[ind-fitting_points:ind+fitting_points] - expected)**2)/std**2)


    def _linear_fit(self, ran=[]):
        if not ran:
            λ = self.λ 
            f = self.force_Y
        else:
            λ = self.λ[min(ran):max(ran)]
            f = self.force_Y[min(ran):max(ran)]
        popt, _ = curve_fit(lambda x, a, b: x*a+b, λ, f)
        return popt[0]*λ+popt[1]


    def _errors(self):
        self.index, self.λ_0, self.f_rupture, self.f_rupture_next, self.x_ssDNA, self.N_nucleotides = [[0]]*6
        # t_0 now represents the initial time of the data
        self.t_0 = [min([self.time[0], self.time[-1]])]
        self.k_F, self.k_U = [0], [0]
        self.params = self.f_rupture + self.f_rupture_next + self.x_ssDNA + self.N_nucleotides + self.t_0 + self.λ_0 + self.k_F + self.k_U + [self.N_fits]

    

    def _set_initial_time(self):
        # check if there is a meta-file:
        self.metafile_path = f'{self.working_dir}/{self.number}/meta_t0.txt'
        if os.path.isfile(self.metafile_path):
            # the file exists
            t_0 = np.loadtxt(self.metafile_path)
        else:
            # create the file
            t_0 = self.time[0] if self.ty == 'u' else self.time[-1]  # start from t = 0 s
            np.savetxt(self.metafile_path, np.array([t_0]), header=f'Molecule: {self.number}')
        return t_0


    def reshape(self, n_points, way='median'):
        num_segments = len(self.force_Y) // n_points

        # Reshape the data into segments of n_points points
        force_Y_reshaped = self.force_Y[:num_segments*n_points].reshape(num_segments, n_points)
        λ_reshaped = self.λ[:num_segments*n_points].reshape(num_segments, n_points)
        time_reshaped = self.time[:num_segments*n_points].reshape(num_segments, n_points)

        # Calculate the average of each segment: mean filter
        if way == 'mean':
            force_Y_reshaped = np.mean(force_Y_reshaped, axis=1)
            λ_reshaped = np.mean(λ_reshaped, axis=1)
            time_reshaped = np.mean(time_reshaped, axis=1)
            return force_Y_reshaped, λ_reshaped, time_reshaped

        # Calculate the median of each segment: median filter
        force_Y_reshaped = np.median(force_Y_reshaped, axis=1)
        λ_reshaped = np.median(λ_reshaped, axis=1)
        time_reshaped = np.median(time_reshaped, axis=1)

        return force_Y_reshaped, λ_reshaped, time_reshaped


    def _calculation_x_d(self, f):
        return self.d*(1./np.tanh((f*self.d)/self.KBT) - self.KBT/(f*self.d)) # nm
    

    # Inverse function of f(x) from WLC model
    def x_WLC_f(self, f):
        # y = x/L

        fnorm = ((4*self.P)/self.KBT)*f
        a = 4
        b = -9-fnorm
        c = 6+2*fnorm
        d = -fnorm

        p = c/a - (b**2)/(3*a**2)
        q = d/a - b*c/(3*a**2) + 2*b**3/(27*a**3)
        
        D = (q**2)/4 + (p**3)/27
        
        if D > 0:
            # In this case, there is only one real root, given by "out" below
            u = np.cbrt(-q/2+np.sqrt(D))
            v = np.cbrt(-q/2-np.sqrt(D))
            out = -b/(3*a)+u+v
        elif D < 0:
            # In this case there 3 real distinct solutions, given by out1,
            # out2, out3 below. The one that interests us is that in the
            # inerval [0,1]. It is seen ("empirically") that is always the
            # second one in the list below [there is perhaps more to search here]
            
            rho = np.sqrt((q**2)/4-D)
            theta = np.arccos(-q/(2*rho))
            out1 = 2*np.sqrt(-p/3)*np.cos(theta/3)-b/(3*a)
            out2 = 2*np.sqrt(-p/3)*np.cos((theta+2*np.pi)/3)-b/(3*a)
            out3 = 2*np.sqrt(-p/3)*np.cos((theta+4*np.pi)/3)-b/(3*a)
            
            # We implement the following check just to be sure out2 is the good root 
            # (in case this "empirical" truth turns out to stop working) 
            try:
                out2 < 0 or out2 > 1
            except:    
                print("The default root doesn't seem to be the good one - you may want to check if the others lie in the interval [0,1]")
            else:
                out = out1
        else:
            # In theory we always go from D>0 to D<0 by passing to a D=0
            # boundary, where we have two real roots (and where the formulas
            # above change again slightly). In practice, however, due to round-off errors,
            # it seems we never hit this boundary but always pass "through" it 
            # This D=0 scenario could still be implemented if needed, though.
            out1 = -2*np.cbrt(q/2)
            out2 = 2*np.cbrt(q/2)   # = out3
            if out1 > 0:
                out = out1
            else:
                out = out2

        y = out
        # L = N*d_aa   # nm

        return y    #*self.L   
    



    def _createTxt(self):
        alphabet = string.ascii_uppercase
        files_of_interest = [f for f in os.listdir(self.dir_name) if self.name in f and 'COM' not in f]
        molecules = set([int(f.split('pull')[0]) for f in files_of_interest])
        for number in molecules:
            os.makedirs(f'{self.working_dir}/{number}')
            name = f'{number}pull{self.f_MAX}A.txt'
            path = self.dir_name + name
            
            # The molecule exists: it's time to merge all the files
            try:
                file = np.loadtxt(path)
            except:
                print(f'Unable to load file in {path}')
                continue

            for i, letter in enumerate(alphabet):
                path = f'{path.split(letter)[0]}{alphabet[i+1]}'
                if os.path.isfile(path): # There is more than a single file
                    new_file = np.loadtxt(path)
                    file = np.concatenate((file, new_file)) # shape = (N, 33) N >> 1
                else:
                    # Change molecule
                    break

            # The whole file is imported
            # Filtering columns...
            λ = (file[:, 25] + file[:, 27])/2
        
            # Columns: CycleCount, X_force,	Y_force, Z_force, time(sec), Status, λ
            file = file[:, [0, 15, 18, 21, 31, 32]]
            file = np.c_[file, λ]
            
            # Select folded & unfolded on the basis of status
            # U for Unfolding (Status=131), F for Folding (Status=130)
            unfolding = file[file[:, -2] == 131]
            folding = file[file[:, -2] == 130]

            save_dir = self.working_dir + f'/{number}/'

            # Differentiate folding and unfolding cycles
            wh_U = np.where(np.diff(unfolding[:, -3]) >= 1)[0]
            wh_U = np.append(wh_U+1, unfolding.shape[0])
            wh_F = np.where(np.diff(folding[:, -3]) >= 1)[0]
            wh_F = np.append(wh_F+1, folding.shape[0])

            # Save data in .txt files
            for i, el in enumerate(wh_U):
                ind = 0 if i == 0 else wh_U[i-1]
                np.savetxt(save_dir + f'{i+1}_u.txt', unfolding[ind:el])

            for j, el in enumerate(wh_F):
                ind = 0 if j == 0 else wh_F[j-1]
                np.savetxt(save_dir + f'{j+1}_f.txt', folding[ind:el])


    def _save_results(self, molecules, all_molecules_f, all_molecules_u):
        # Columns: Molecule, f, f_next, x_ssDNA, N_nucleotides, t_0
        self.res_fold = pd.DataFrame(columns=["Molecule", "f", "f_next", "x_ssDNA", "N_nucleotides", "t_0"])
        self.res_unfold = pd.DataFrame(columns=["Molecule", "f", "f_next", "x_ssDNA", "N_nucleotides", "t_0"])
        j = 0
        for m in range(len(molecules)):
            for i in range(len(all_molecules_f[m])):
                self.res_fold.loc[i+j] = [molecules[m]]+all_molecules_f[m][i][0]
            j += len(all_molecules_f[m])

        j = 0
        for m in range(len(molecules)):
            for i in range(len(all_molecules_u[m])):
                self.res_unfold.loc[i+j] = [molecules[m]]+all_molecules_u[m][i][0]
            j += len(all_molecules_f[m])

        results_folder = f'res/{self.folder}'
        if not os.path.exists(results_folder):
            os.makedirs(results_folder) # create folder
    
        result_path_folding = f'res/{self.folder}/f_max{self.f_MAX}_folding.txt'
        result_path_unfolding = f'res/{self.folder}/f_max{self.f_MAX}_unfolding.txt'

        print("Saving results...") # create files
        np.savetxt(result_path_folding, self.res_fold.values)
        np.savetxt(result_path_unfolding, self.res_unfold.values)


    def make_plot(self, save_fig, folder, number, N, txt=True):
        fig = plt.figure()
        plt.plot(self.λ_original[:self.filter_10], self.force_Y, label = 'Data')
        if self.N_fits == 2:
            ind = int(self.index[0])
            fitting_points = self.fitting_points if ind > self.fitting_points else ind
            plt.plot(self.λ_original[:self.filter_10][ind-fitting_points:ind+fitting_points], 
                     self.theor_f[ind-fitting_points:ind+fitting_points], lw=2.5, label = 'Fit')
        plt.xlabel('$\lambda \\: [nm]$')
        plt.ylabel('$f_Y \\: [pN]$')
        plt.grid()
        plt.legend()
        plt.title(f'{self.path}')
        if txt:
            plt.text(x=min(self.λ), y=min([max(self.force_Y)/2, int(self.f_MAX)/2]), s=f'N = {self.N_nucleotides}\n # of fits = {self.N_fits}\n f_r = {self.f_rupture}')
        if save_fig:
            path = f'imgs/{folder}/{self.folder}/{self.f_MAX}'
            if not os.path.exists(path):
                os.makedirs(path) # create folder
            plt.savefig(f'{path}/{number}_{N}_{self.ty}.png', dpi=100)
            plt.close(fig) 
        else:
            plt.show()
