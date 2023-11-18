import os
import numpy as np
import string
from scipy.optimize import curve_fit
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import matplotlib.pyplot as plt

class Txt_Reading:
    def __init__(self, folder, f_MAX) -> None:
        self.KBT = 4.11     # pN*nm
        self.d = 2          # nm
        self.P = 1.35       # ± 0.05 nm -> persistence length
        self.d_aa = 0.58    # ± 0.02 nm/base -> distance between consecutive nucleotides
        self.loading_rate = 6   # pN/s -> useful to discard useless data. This value can be changed to 5 or 4.5
        self.folder = folder
        # self.dir_name = 'G:/Il mio Drive/Fisica/Dispense Università/2 anno/Zaltron & Xavi/' + dir_name
        self.dir_name = 'F:/Zaltron & Xavi/' + self.folder + '/'
        self.f_MAX = str(f_MAX)
        self.name = f'pull{self.f_MAX}'
        self.dictionary = {'10':9, '15':9, '20':9, '25':9, '30':9, '35':9}
        self.fit = {1:15, 2:15, 3:15, 4:10, 5:10, 6:7, 7:7, 8:5, 9:5, 10:10}
        self.decision = 3.0 if int(self.f_MAX) < 15 else -0.2

        self.working_dir = f'Data/{self.folder}/{self.f_MAX}'
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            self._createTxt()
        

    
    def readTxt(self, number = 1, N = 1, ty = 'u', print_out = True, forced_reshaped = 0, graph = False, initial_t_time = False):
        self.number = number    # molecule number
        self.ty = ty    # file type: unfolding (u) or folding (f) 
        self.path = f'{self.working_dir}/{self.number}/{N}_{self.ty}.txt'

        self.file = np.loadtxt(self.path)

        # take account only of rows with f = 100 kHz
        self.file = self.file[1:][np.diff(self.file[:, 0]) == 1, :]

        self.λ_original = self.file[:, -1] if ty == 'u' else self.file[:, -1][::-1]
        self.force_Y = np.abs(self.file[:, 2]) if ty == 'u' else np.abs(self.file[:, 2][::-1])
        self.time = self.file[:, -3] if ty == 'u' else self.file[:, -3][::-1]
        self.t_initial = self._set_initial_time() if initial_t_time else (self.time[0] if ty == 'u' else self.time[-1])
        self.time -= self.t_initial # different initial time for each file, unique for each molecule

        # Correction just to preserve the next part of the analysis if for any chance
        # the x-axis is shifted back, into negatives numbers
        # Normalization
        self.λ = (self.λ_original - np.min(self.λ_original))/(np.max(self.λ_original)-np.min(self.λ_original))

        self.fitting_points = 15  if int(self.f_MAX) < 15 else 10 # number of points used for linear fits
        self.N_fits = 0     # Number of fits performed: it can be 0 (no results), 1 (the molecule doesn't open/close), 2 (standard case)
        self.best, self.linear = [], []

        # if not forced_reshaped:
        #     # default reshape of 5
        #     self.force_Y, self.λ = self.reshape(5)

        try:
            self.analysis(print_out=print_out, forced_reshaped=forced_reshaped, graph=graph)
        except:
            self._errors()


        columns = ["f_rupture", "f_rupture_next", "x_ssDNA", "N_nucleotides", "t_0", "λ_0", "a_pre", "b_pre", "a_post", "b_post", "N_fits"]
        self.params_df = pd.DataFrame([self.params], columns=columns)

        return self.file
    

    def find_jumps(self, force_Y):
        new_force_Y = force_Y[np.where(force_Y>=0)[0].tolist() and np.where(force_Y<=8)[0].tolist()]
        all_indexes = np.array(np.where(np.diff(new_force_Y) < -2.7*np.std(np.diff(force_Y)))[0].tolist() + np.where(np.diff(new_force_Y) > 2.7*np.std(np.diff(force_Y)))[0].tolist())
        # l = int(len(force_Y) // 4.5)
        # f = int(len(force_Y) // 1.5) 
        # prova = all_indexes[all_indexes > l]
        # prova = prova[prova < f]
        all_indexes = all_indexes[all_indexes > 5]
        prova = []
        if len(prova) != 0:
            all_indexes = prova.tolist()
        else:
            all_indexes = all_indexes.tolist()

        if len(all_indexes) % 2 == 0 and len(all_indexes) != 0:
            all_indexes.append(0)
            median_index = np.array([np.median(all_indexes)], dtype=int)
        else:
            median_index = [int(self.force_Y.size/2)]
        return all_indexes, median_index


    def analysis(self, print_out=True, forced_reshaped=False, graph=False):
        if np.abs(self.time[-1] - self.time[0]) < (np.max(self.force_Y) - np.min(self.force_Y)
                                                   )/self.loading_rate or max(self.force_Y)+1 < int(self.f_MAX
                                                                                                  ) or min(self.force_Y) > int(self.f_MAX
                                                                                                                               )/3 or self.λ.size < 50 or self.force_Y[-50:-1].mean() < int(self.f_MAX)/2 or max(self.force_Y) > int(self.f_MAX
                                                                                                                                                                                                                                     )+4 or self.λ[np.where(self.force_Y == min(
                                                                                                                                                                                                                                         self.force_Y))[0][0]]>0.1:
            self._errors()
            # Noise or other useless data
            if graph:
                self.make_plot(save_fig=False, folder=self.folder, number=self.number, N=self.path.split('_')[0].split('/')[-1], txt=False)
            return -1

        force_Y_reshaped, λ_reshaped, time_reshaped = self.reshape(5)
        self.all_indexes, self.index_med = self.find_jumps(force_Y=force_Y_reshaped)
        self.index, c, N_p = self.bootstrap(λ_reshaped, force_Y_reshaped, time_reshaped, self.all_indexes, self.index_med[0])
        self.index = np.array([self.index])
        
        if self.index[0] < 5:
            # it's impossible to perform a fit: try changing the index
            # and see if reshaping the problem solves itself
            self.index = [6]
        
        if force_Y_reshaped[self.index] > 7:
            self.linear.append(1)
        elif c != np.Inf and force_Y_reshaped[self.index] > 3 and N_p < 80:
            self.best.append([1, c, N_p])


        if forced_reshaped:
            self.λ_0, self.index, self.f_rupture = self.change_point(forced_reshaped=forced_reshaped)
            self.N_fits = 2
            self.params = self.f_rupture + self.f_rupture_next + self.x_ssDNA + self.N_nucleotides + self.t_0 + self.λ_0 + self.popt_pre.tolist() + self.popt_post.tolist() + [self.N_fits]
            self.theor_f = self._heviside_fitting(self.λ, *self.params[5:-1])   

        else: 
            self.λ_0 = λ_reshaped[self.index].tolist()
            self.f_rupture = force_Y_reshaped[self.index].tolist()
            # self.f_rupture_next = force_Y_reshaped[self.index+1].tolist()
            # new fits, with λ_0 position fixed
            ind = int(self.index[0])
            fitting_points = self.fitting_points if ind > self.fitting_points else ind
            
            self.popt_pre, self.popt_post = self._compute_fit(ind, fitting_points, λ_reshaped, force_Y_reshaped)
            self.f_rupture_next = [(self.popt_post[0]*self.λ[ind:ind+fitting_points] + self.popt_post[1])[0]]
            # print(self.f_rupture, self.f_rupture_next)
            
            self.t_0, self.k_eff, self.x_ssDNA, self.N_nucleotides = self._compute_interesting_variables(time_reshaped, self.popt_pre, ind, self.f_rupture, self.f_rupture_next)
            
            # We expect N of nucleotides > 30 (if the molecule opens)
            # If ty == 'u', the difference between the rupture force and the next one will be, hopefully, positive,
            # while for ty == 'f' it should be negative, since data are taken in the "opposite" direction:
            # self.condition = self.f_rupture[0] - self.f_rupture_next[0] < 0.1 if self.ty == 'u' else self.f_rupture_next[0] - self.f_rupture[0] < 0.1
            
            if (self.f_rupture[0] - self.f_rupture_next[0] < 0.1) or self.λ_0[0] > 0.51 or self.λ_0[0] < 0.12 or self.N_nucleotides[0] < 30 or self.N_nucleotides[0] > 60: 
                if print_out:
                    print(f'The break point λ_0 {self.λ_0} could be smaller/higher than expected')
                    print(f'or the rupture force {self.f_rupture} smaller/higher than expected') 
                self.perform_reshapes()

            else:
                self.force_Y, self.λ, self.time = force_Y_reshaped, λ_reshaped, time_reshaped                

            if self.linear and not self.best:
                # # First: try another way to reshape data:
                # force_Y_reshaped, λ_reshaped, time_reshaped = self.reshape(n_points=5, way='mean')
                # all_indexes, index_med = self.find_jumps(force_Y=force_Y_reshaped)
                # index, _ = self.bootstrap(λ_reshaped, force_Y_reshaped, all_indexes)
                # self.index = np.array([index])
                # self.λ_0 = λ_reshaped[self.index].tolist()
                # self.f_rupture = force_Y_reshaped[self.index].tolist()
                # ind = int(self.index[0])
                # fitting_points = self.fitting_points if ind > self.fitting_points else ind
                
                # self.popt_pre, self.popt_post = self._compute_fit(ind, fitting_points, λ_reshaped, force_Y_reshaped)
                # self.f_rupture_next = [(self.popt_post[0]*self.λ[ind:ind+fitting_points] + self.popt_post[1])[0]]

                # self.t_0, self.k_eff, self.x_ssDNA, self.N_nucleotides = self._compute_interesting_variables(time_reshaped, self.popt_pre, ind, self.f_rupture, self.f_rupture_next)

                # if (self.f_rupture[0] - self.f_rupture_next[0] < 0.1) or self.λ_0[0] > 0.51 or self.λ_0[0] < 0.12 or self.N_nucleotides[0] < 30 or self.N_nucleotides[0] > 60: 
                
                # It's impossible to perform a fit: return 0
                # This means that a single linear fit could explain the data
                # No need to perform a double linear fit
                self.N_fits = 1
                self._errors()
                self.theor_f = self._linear_fit()
                
                # else:
                #     self.params = self.f_rupture + self.f_rupture_next + self.x_ssDNA + self.N_nucleotides + self.t_0 + self.λ_0 + self.popt_pre.tolist() + self.popt_post.tolist() + [self.N_fits]
                #     self.theor_f = self._heviside_fitting(self.λ, *self.params[5:-1])

            elif (not self.linear and self.best) or (not self.linear and not self.best):
                self.N_fits = 2
                if self.best:
                    b = self.choice(np.array(self.best).flatten())
                    # b = 5*b if b != 1 else b
                    self.λ_0, self.index, self.f_rupture = self.change_point(forced_reshaped=b*5)  
                # else:
                #     self.λ_0, self.index, self.f_rupture = self.change_point(forced_reshaped=1)  
                self.params = self.f_rupture + self.f_rupture_next + self.x_ssDNA + self.N_nucleotides + self.t_0 + self.λ_0 + self.popt_pre.tolist() + self.popt_post.tolist() + [self.N_fits]
                self.theor_f = self._heviside_fitting(self.λ, *self.params[5:-1])

            elif self.linear and self.best:
                # both linear and double-jump fits had been found
                # compute χ² to evaluate best result
                b = self.choice(np.array(self.best).flatten())
                self.λ_0, self.index, self.f_rupture = self.change_point(forced_reshaped=b*5)   
                if int(self.f_MAX) < 15: 
                    chi_models = np.array([self.chi_squared(mode='linear')/len(self.linear), self.chi_squared()/len(self.best)])
                else:
                    chi_models = np.array([self.chi_squared(mode='linear'), self.chi_squared()])    
                if print_out:
                    print(f'χ² = {chi_models}')
                
                wh = np.where(chi_models == min(chi_models))[0]

                if np.abs(np.diff(chi_models)) <= self.decision or wh[0] == 1 or chi_models[1] < 10:
                    # double-jump
                    self.N_fits = 2
                    self.params = self.f_rupture + self.f_rupture_next + self.x_ssDNA + self.N_nucleotides + self.t_0 + self.λ_0 + self.popt_pre.tolist() + self.popt_post.tolist() + [self.N_fits]
                    self.theor_f = self._heviside_fitting(self.λ, *self.params[5:-1])   
                else:
                    # linear
                    self.N_fits = 1
                    self._errors()
                    self.theor_f = self._linear_fit()

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


    def _compute_interesting_variables(self, time, popt_pre, index, f_rupture, f_rupture_next):
        # This function computes important variables such as:
        # t_0: when the jump happens
        # k_eff: effective k
        # x_ssDNA: length of the single strand DNA
        # N_nucleotides: Number of nucleoteotides
        t_0 = [time[index]]
        k_eff = popt_pre[0]/(np.max(self.λ_original)-np.min(self.λ_original)) 
        x_ssDNA = np.array([np.abs(f_rupture[0]-f_rupture_next[0])/k_eff + self._calculation_x_d(f_rupture[0])]).tolist()
        N_nucleotides = np.array([x_ssDNA[0] / (self.x_WLC_f(f_rupture[0]) * self.d_aa)]).tolist()
        return t_0, k_eff, x_ssDNA, N_nucleotides
        

    def bootstrap(self, λ_reshaped, force_Y_reshaped, time_reshaped, all_indexes, index_med):
        best_index, best_chi = 6, np.Inf
        N_prev = 0
        for index in all_indexes:
            if index < 5:
                continue
            λ_0 = λ_reshaped[index]
            f_rupture, f_rupture_next = force_Y_reshaped[index:index+2]
            fitting_points = self.fitting_points if index > self.fitting_points else index

            try:
                popt_pre, popt_post = self._compute_fit(index, fitting_points, λ_reshaped, force_Y_reshaped)
                f_rupture_next = (popt_post[0]*λ_reshaped[index:index+fitting_points] + popt_post[1])[0]
                t_0, k_eff, x_ssDNA, N_nucleotides = self._compute_interesting_variables(time_reshaped, popt_pre, index, [f_rupture], [f_rupture_next])
            except:       
                continue

            # The fit has been performed: we have to find a way to evaluate the best solution among all the possible results:
            # χ² could be a reasonable option
            [a, b], [c, d] = popt_pre, popt_post
            # Pearson formulation
            expected = self._heviside_fitting(λ_reshaped[index-fitting_points:index+fitting_points], λ_0, a, b, c, d)
            std = np.std(force_Y_reshaped[index-fitting_points:index+fitting_points])
            chi = np.sum(((force_Y_reshaped[index-fitting_points:index+fitting_points] - expected)/std)**2)
            # print(f'The index is {index} \n The chi^2 is {chi}')
            if chi < best_chi and f_rupture > f_rupture_next and N_nucleotides[0] > N_prev:
                best_chi = chi
                best_index = index
                N_prev = N_nucleotides[0]

        return best_index, best_chi, N_prev


    def change_point(self, forced_reshaped=0):
        self.force_Y, self.λ, self.time = self.reshape(n_points=forced_reshaped)
        all_indexes, index_med = self.find_jumps(self.force_Y)
        # print(f'all indexes = {all_indexes}')
        index, _, _ = self.bootstrap(λ_reshaped=self.λ, force_Y_reshaped=self.force_Y, time_reshaped=self.time, all_indexes=all_indexes, index_med=index_med[0])
        # print(index)
        # if index < 5:
        #     print(f'Index is {index}')
        #     return [np.Inf]*3
        fitting_points = self.fitting_points if index > self.fitting_points else index
        # print(f'fitting points = {fitting_points}')

        λ_0 = self.λ[index]
        f_rupture = self.force_Y[index]
        popt_pre, popt_post = self._compute_fit(index, fitting_points, self.λ, self.force_Y)
        f_rupture_next = (popt_post[0]*self.λ[index:index+fitting_points] + popt_post[1])[0]
            
        self.t_0, self.k_eff, self.x_ssDNA, self.N_nucleotides = self._compute_interesting_variables(self.time, popt_pre, index, [f_rupture], [f_rupture_next])

        self.f_rupture_next = [f_rupture_next]
        self.fitting_points = fitting_points
        self.popt_pre, self.popt_post = popt_pre, popt_post
        # self.best, self.linear = [], []
        
        return [λ_0], np.array([index]), [f_rupture]


    def perform_reshapes(self):
        # The aim of this function is to find the best λ_0, if the differrence method doesn't work
        # The all_indexes array has been ysed yet, so it's time to use those points to perform kind
        # of a bootstrap method in order to find the best rip point.
        # Attention has to be done since the higher the concentration of oligoneuclites the lower
        # the λ_0 point will be: so the point spotted here could not be the real one
        
        segments = list(range(2, self.dictionary[self.f_MAX]))
        off_chi = np.Inf
        for n_points in segments: 
            force_Y_reshaped, λ_reshaped, time_reshaped = self.reshape(n_points=n_points*5)

            # The index of the λ_0 in the λ array is where the difference of consecutive forces has minimum
            # times the scale factor - n_points - used to calculate the average before  
            
            all_indexes, index_med = self.find_jumps(force_Y=force_Y_reshaped)
            if index_med == None:
                continue
            # for this configuration we aim to find the best possible index, among all the available ones.
            ind, chi, N_prev = self.bootstrap(λ_reshaped=λ_reshaped, force_Y_reshaped=force_Y_reshaped, time_reshaped=time_reshaped, all_indexes=all_indexes, index_med=index_med[0])            
            λ_0 = λ_reshaped[ind]

            f_rupture = force_Y_reshaped[ind]
            fitting_points = self.fitting_points if ind > self.fitting_points else ind
            # fitting_points = self.fit[n_points] if ind > self.fit[n_points] else ind
            
            try:
                popt_pre, popt_post = self._compute_fit(ind, fitting_points, λ_reshaped, force_Y_reshaped)
            except:
                continue
            # t_0, k_eff, x_ssDNA, N_nucleotides = self._compute_interesting_variables(popt_pre, ind, [f_rupture], [f_rupture_next])
            # index = np.array([ind])
            f_rupture_next = (popt_post[0]*λ_reshaped[ind:ind+fitting_points] + popt_post[1])[0]
            
            # print(f'Reshape = {n_points} \n f_rupture = {f_rupture} \n f_rupture_next = {f_rupture_next} \n index = {ind}')
            
            # condition = f_rupture - f_rupture_next < 0.1 if self.ty == 'u' else f_rupture_next - f_rupture < 0.1
            cond = True if int(self.f_MAX) < 15 else f_rupture > 3 
            if f_rupture - f_rupture_next > 0.1 and cond and N_prev < 80: # and (0.12 < λ_0 < 0.51): 
                # work with reshaped data
                if chi < off_chi:
                    off_chi = chi
                    self.best.append([n_points, chi, N_prev])

                else:
                    self.linear.append(n_points)

                # self.force_Y = force_Y_reshaped.copy()
                # self.λ = λ_reshaped.copy()
                # self.f_rupture_next = [f_rupture_next]
                # self.fitting_points = fitting_points
                # self.popt_pre, self.popt_post = popt_pre, popt_post
                # self.t_0, self.k_eff, self.x_ssDNA, self.N_nucleotides = t_0, k_eff, x_ssDNA, N_nucleotides
                # self.best, self.linear = [], []
                # # if any improvement
                # if print_out:
                #     print(f'Reshape of {n_points*5} performed')
                # return [λ_0], index, [f_rupture]
            
            else:
                # # save the best result found: at least an improvement
                # if (f_rupture > int(self.f_MAX) or λ_0 < 0.12 or 10 < N_nucleotides[0] < 30):
                self.linear.append(n_points)
                # elif 0 < N_nucleotides[0] < self.N_nucleotides[0] and (f_rupture - f_rupture_next > 0.1):
                #     self.best.append(n_points)
        # else, get back the initial result found (default reshape of 5)
        # return self.λ_0, self.index, self.f_rupture
                
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
    
    def choice(self, best):
        n = best[::3]
        chi = best[1::3]
        N_prev = best[2::3]
        lowest_chi = np.where(np.array(chi) == np.min(chi))[0][0]
        highest_N = np.where(np.array(N_prev) == np.max(N_prev))[0][0]
        # If they coincide, ok
        if lowest_chi == highest_N:
            return int(n[lowest_chi])
        else:
            # They are different, so choose wisely:
            if 25 < np.max(N_prev) < 60:
                return int(n[highest_N])
            else:
                return int(n[lowest_chi])

    

    def sequential_analysis(self, print_not_saved=True, save_files=True):
        molecules = sorted([int(m) for m in os.listdir(self.working_dir)])  # molecule numbers
        # Each molecule has different folding and unfolding cycles
        all_molecules_f = []
        all_molecules_u = []
        for m in tqdm(molecules):
            m_f = []
            m_u = []
            path = f'{self.working_dir}/{m}/'
            all_files = np.array(os.listdir(path))
            fold_N_max = max([int(f.split("_f.txt")[0]) for f in all_files if "_f.txt" in f])
            unfold_N_max = max([int(f.split("_u.txt")[0]) for f in all_files if "_u.txt" in f])
            for N in range(1, fold_N_max+1):
                file_f = self.readTxt(number = m, N = N, ty = 'f', print_out=False, initial_t_time=True)
                if self.N_fits > 0 and 0 <= self.N_nucleotides[0] < 100:
                    m_f.append([self.params[:5]]) # saving the parameters
                    if self.N_fits == 1:
                        self.make_plot(save_fig=True, folder='saved/linear', number=m, N=N)
                    else:
                        self.make_plot(save_fig=True, folder='saved', number=m, N=N)
                else:
                    if print_not_saved:
                        print(f'Not saving file {self.path}')
                        if len(self.λ) != 0:
                            self.make_plot(save_fig=True, folder='not_saved', number=m, N=N)
                        
            for N in range(1, unfold_N_max+1):
                file_u = self.readTxt(number = m, N = N, ty = 'u', print_out=False, initial_t_time=True)
                if self.N_fits > 0 and 0 <= self.N_nucleotides[0] < 100:
                    m_u.append([self.params[:5]]) # saving the parameters 
                    if self.N_fits == 1:
                        self.make_plot(save_fig=True, folder='saved/linear', number=m, N=N)
                    else:
                        self.make_plot(save_fig=True, folder='saved', number=m, N=N)
                else:
                    if print_not_saved:
                        print(f'Not saving file {self.path}')
                        if len(self.λ) != 0:
                            self.make_plot(save_fig=True, folder='not_saved', number=m, N=N)
                        
            all_molecules_f.append(m_f)
            all_molecules_u.append(m_u)

        if save_files:
            self._save_results(molecules, all_molecules_f, all_molecules_u)

        return molecules, all_molecules_f, all_molecules_u

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
            wh_U = np.append(wh_U, unfolding.shape[0])
            wh_F = np.where(np.diff(folding[:, -3]) >= 1)[0]
            wh_F = np.append(wh_F, folding.shape[0])

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

  
    def _heviside_fitting(self, x, λ_0, a, b, c, d):
        return (x*a + b) * np.heaviside(λ_0 - x, 0.5) + (c*x + d) * np.heaviside(x - λ_0, 0.5) 
    
    def _calculation_x_d(self, f):
        return self.d*(1./np.tanh((f*self.d)/self.KBT) - self.KBT/(f*self.d)) # nm
    
  
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
    

    def _errors(self):
        self.index, self.λ_0, self.f_rupture, self.f_rupture_next, self.x_ssDNA, self.N_nucleotides = [[0]]*6
        # t_0 now represents the initial time of the data
        self.t_0 = [min([self.time[0], self.time[-1]])]
        self.popt_pre, self.popt_post = [0, 0], [0, 0]
        self.params = self.f_rupture + self.f_rupture_next + self.x_ssDNA + self.N_nucleotides + self.t_0 + self.λ_0 + self.popt_pre + self.popt_post + [self.N_fits]

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
    

    def _compute_fit(self, ind, fitting_points, λ, force_Y):
        pre_λ = λ[ind-fitting_points:ind]
        pre_f = force_Y[ind-fitting_points:ind]
        popt_pre, _ = curve_fit(lambda x, a, b: x*a+b, pre_λ, pre_f)
        post_λ = λ[ind:ind+fitting_points]
        post_f = force_Y[ind:ind+fitting_points]
        popt_post, _ = curve_fit(lambda x, a, b: x*a+b, post_λ, post_f)
        return popt_pre, popt_post
    
    def _linear_fit(self, ran=[]):
        if not ran:
            λ = self.λ 
            f = self.force_Y
        else:
            λ = self.λ[min(ran):max(ran)]
            f = self.force_Y[min(ran):max(ran)]
        popt, _ = curve_fit(lambda x, a, b: x*a+b, λ, f)
        return popt[0]*λ+popt[1]
        
    def chi_squared(self, index=0, mode=""):
        ind = self.index[0] if not index else index
        fitting_points = self.fitting_points if ind > self.fitting_points else ind
        std = np.std(self.force_Y[ind-fitting_points:ind+fitting_points])
        if mode:
            # linear
            expected = self._linear_fit(ran=[ind-fitting_points, ind+fitting_points])
            return (np.sum(((self.force_Y[ind-fitting_points:ind+fitting_points] - expected)**2)/std**2))
        # default: double-jump
        [a, b], [c, d] = self._compute_fit(ind=ind, fitting_points=fitting_points, λ=self.λ, force_Y=self.force_Y)
        expected = self._heviside_fitting(self.λ[ind-fitting_points:ind+fitting_points], self.λ_0, a, b, c, d)
        return np.sum(((self.force_Y[ind-fitting_points:ind+fitting_points] - expected)**2)/std**2)

    def make_plot(self, save_fig, folder, number, N, txt=True):
        fig = plt.figure()
        plt.plot(self.λ, self.force_Y, label = 'Data')
        if self.N_fits == 2:
            ind = int(self.index[0])
            fitting_points = self.fitting_points if ind > self.fitting_points else ind
            plt.plot(self.λ[ind-fitting_points:ind+fitting_points], self.theor_f[ind-fitting_points:ind+fitting_points], lw=2.5, label = 'Fit')
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
