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
        self.folder = folder
        # self.dir_name = 'G:/Il mio Drive/Fisica/Dispense Università/2 anno/Zaltron & Xavi/' + dir_name
        self.dir_name = 'F:/Zaltron & Xavi/' + self.folder + '/'
        self.f_MAX = str(f_MAX)
        self.name = f'pull{self.f_MAX}'

        self.working_dir = f'Data/{self.folder}/{self.f_MAX}'
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            self._createTxt()
        

    
    def readTxt(self, number = 1, N = 1, ty = 'u', print_out=True, graph = False):
        self.number = number    # molecule number
        self.path = f'{self.working_dir}/{self.number}/{N}_{ty}.txt'

        self.file = np.loadtxt(self.path)

        # take account only of rows with f = 100 kHz
        self.file = self.file[1:][np.diff(self.file[:, 0]) == 1, :]

        self.λ = self.file[:, -1] if ty == 'u' else self.file[:, -1][::-1]
        self.force_Y = np.abs(self.file[:, 2]) if ty == 'u' else np.abs(self.file[:, 2][::-1])
        self.time = self.file[:, -3]
        self.time -= self.time[0] # start from t = 0 s

        # Correction just to preserve the next part of the analysis if for any chance
        # the x-axis is shifted back, into negatives numbers
        if np.min(self.λ) < 0:
            self.λ = self.λ - np.min(self.λ) + 200  ## ??

        if np.max(self.λ) < 1000:
            self.analysis(print_out, graph)
        else:
            self.params = np.zeros(10) # 10 = number of parameters
            self.index, self.λ_0, self.f_rupture, self.t_0 = [None]*4

        columns = ["f_rupture", "f_rupture_next", "x_ssDNA", "N_nucleotides", "t_0", "λ_0", "a_pre", "b_pre", "a_post", "b_post"]
        self.params_df = pd.DataFrame([self.params], columns=columns)

        return self.file
    
    def analysis(self, print_out=True, graph = False):
        self.index = np.array([np.where(np.diff(self.force_Y) == min(np.diff(self.force_Y)))[0][0]])
        self.λ_0 = self.λ[self.index].tolist()
        self.f_rupture = self.force_Y[self.index].tolist()

        if self.f_rupture[0] > 7 or self.f_rupture[0] < 2.9: 
            if print_out:
                print(f'The break point λ_0 {self.λ_0} could be smaller/higher than expected')
                print(f'or the rupture force {self.f_rupture} smaller/higher than expected')
            self.λ_0, self.index, self.f_rupture = self.change_point(print_out)

        self.f_rupture_next = self.force_Y[self.index + 1].tolist()
        # new fits, with λ_0 position fixed
        pre_λ = self.λ[:self.index[0]]
        pre_f = self.force_Y[:self.index[0]]
        popt_pre, _ = curve_fit(lambda x, a, b: x*a+b, pre_λ, pre_f)
        post_λ = self.λ[self.index[0]:]
        post_f = self.force_Y[self.index[0]:]
        popt_post, _ = curve_fit(lambda x, a, b: x*a+b, post_λ, post_f)
        self.t_0 = self.time[self.index].tolist()
        self.k_eff = popt_pre[0] # np.abs((self.f_rupture - self.force_Y[0])/(self.λ_0 - self.λ[0]))
        self.x_ssDNA = np.array([np.abs(self.f_rupture[0]-self.f_rupture_next[0])/self.k_eff + self._calculation_x_d(self.f_rupture[0])]).tolist()
        self.N_nucleotides = np.array([self.x_ssDNA[0] / (self.x_WLC_f(self.f_rupture[0]) * self.d_aa)]).tolist()

        self.params = self.f_rupture + self.f_rupture_next + self.x_ssDNA + self.N_nucleotides + self.t_0 + self.λ_0 + popt_pre.tolist() + popt_post.tolist()
        theor_f = self._heviside_fitting(self.λ, *self.params[5:])

        if graph:
            plt.plot(self.λ, self.force_Y, label = 'Data')
            plt.plot(self.λ, theor_f, label = 'Fit')
            plt.grid()
            plt.legend()
            plt.title(f'{self.path}')
            plt.show()        



    def change_point(self, print_out=True):
        # The aim of this function is to find the best λ_0 is the differrence method doesn't work
        # Attention has to be done since the higher the concentration of oligoneuclites the lower
        # the λ_0 point will be: so the point spotted here could not be the real one
        # Up to now this function just poits out that maybe the point could be a smaller than expected,
        # but doesn't perform any computation
        segments = [5, 10, 15, 20, 30]
        for n_points in segments: 
            num_segments = len(self.force_Y) // n_points

            # Reshape the data into segments of n_points points
            force_Y_reshaped = self.force_Y[:num_segments*n_points].reshape(num_segments, n_points)
            λ_reshaped = self.λ[:num_segments*n_points].reshape(num_segments, n_points)

            # Calculate the average of each segment
            force_Y_reshaped = np.mean(force_Y_reshaped, axis=1)
            λ_reshaped = np.mean(λ_reshaped, axis=1)

            # The index of the λ_0 in the λ array is where the difference of consecutive forces has minimum
            # times the scale factor - n_points - used to calculate the average before  
            index = np.array([np.where(np.diff(force_Y_reshaped) == min(np.diff(force_Y_reshaped)))[0][0]]) * n_points
            λ_0 = self.λ[index]
            f_rupture = self.force_Y[index]

            if 2.9 < f_rupture[0] < 7: 
                # if any (or) improvement
                if print_out:
                    print(f'Reshape of {n_points} performed')
                return λ_0.tolist(), index, f_rupture.tolist()
        # else, get back the previous result
        return self.λ_0, self.index, self.f_rupture
    

    def sequential_analysis(self):
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
                file_f = self.readTxt(number = m, N = N, ty = 'f', print_out=False)
                if 1.5 < self.params[0] < 9.2 and 10 < self.N_nucleotides[0] < 110:
                    m_f.append([self.params[:5]]) # saving the parameters
                else:
                    print(f'Not saving file {self.path}')
            for N in range(1, unfold_N_max+1):
                file_u = self.readTxt(number = m, N = N, ty = 'u', print_out=False)
                if 1.5 < self.params[0] < 9.2 and 10 < self.N_nucleotides[0] < 110:
                    m_u.append([self.params[:5]]) # saving the parameters 
                else:
                    print(f'Not saving file {self.path}')

            all_molecules_f.append(m_f)
            all_molecules_u.append(m_u)

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
            
            # Select folded & unfolden on the basis of status
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

    def _heviside_fitting(self, x, λ_0, a, b, c, d):
        return (x*a + b) * np.heaviside(λ_0 - x, 0.5) + (c*x + d) * np.heaviside(x - λ_0, 0.5) 
    
    def _calculation_x_d(self, f):
        return self.d*(1./np.tanh((f*self.d)/self.KBT) - self.KBT/(f*self.d)) # nm
    
    # Inverse function of f(x) from WLC model
    def x_WLC_f(self, f):
        fnorm = ((4*self.P)/self.KBT)*f
        a2 = (1/4)*(-9-fnorm)
        a1 = (3/2)+(1/2)*fnorm
        a0 = -fnorm/4
        
        R = (9*a1*a2-27*a0-2*a2**3)/54.
        Q = (3*a1-a2**2)/9.
        
        D = Q**3+R**2
        
        if D > 0:
            # In this case, there is only one real root, given by "out" below
            S = np.cbrt(R+np.sqrt(D))
            T = np.cbrt(R-np.sqrt(D))
            out = (-1/3)*a2+S+T
        elif D < 0:
            # In this case there 3 real distinct solutions, given by out1,
            # out2, out3 below. The one that interests us is that in the
            # inerval [0,1]. It is seen ("empirically") that is always the
            # second one in the list below [there is perhaps more to search here]
            
            theta = np.arccos(R/np.sqrt(-Q**3))
            # out1 = 2*np.sqrt(-Q)*np.cos(theta/3)-(1/3)*a2;
            out2 = 2*np.sqrt(-Q)*np.cos((theta+2*np.pi)/3)-(1/3)*a2
            # out3 = 2*np.sqrt(-Q)*np.cos((theta+4*np.pi)/3)-(1/3)*a2
            
            # We implement the following check just to be sure out2 is the good root 
            # (in case this "empirical" truth turns out to stop working) 
            try:
                out2 < 0 or out2 > 1
            except:    
                print('The default root doesn"t seem the be good one - you may want to check if the others lie in the interval [0,1]')
            else:
                out = out2
        else:
            # In theory we always go from D>0 to D<0 by passing to a D=0
            # boundary, where we have two real roots (and where the formulas
            # above change again slightly). In practice, however, due to round-off errors,
            # it seems we never hit this boundary but always pass "through" it 
            # This D=0 scenario could still be implemented if needed, though.
            print('#ToDo')

        z = out
        # L = N*d_aa   # nm

        return z    #*self.L   