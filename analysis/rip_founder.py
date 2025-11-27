import numpy as np
from scipy.optimize import curve_fit
from .draw import Draw
import warnings
warnings.filterwarnings("ignore")

class method2:
    def __init__(self, λ, force_Y, time, method) -> None:
        self.λ = λ
        self.force_Y = force_Y
        self.time = time
        self.reading = method
        # self.main()

        
    def main(self):
        
        """
        Try fitting
        First method involving χ^2 minimization. The `best_state` is the one corresponding to the least χ^2 found 
        among the possible reshapes.
        """
        
        guess_λ_0 = int(self.λ.size/4)
        popt_pre, popt_post = self.fit(self.λ, self.force_Y, guess_λ_0)

        x = np.concatenate((np.linspace(min(self.λ), self.λ[guess_λ_0], guess_λ_0), np.linspace(self.λ[guess_λ_0], max(self.λ), len(self.λ)-guess_λ_0)))
        theor_fit = self._heaviside_fitting(x, guess_λ_0, *popt_pre, *popt_post)
        old_chi = self.chi_squared(self.force_Y, theor_fit, guess_λ_0)


        best_iter = 0
        best_reshape = 1
        best_popt_pre, best_popt_post = popt_pre, popt_post
        max_iter = 5
        best_chi = np.inf
        possible_reshapes = [5, 10, 15, 20, 25, 30]

        guess_λ_0 = 0 
        keep_track = {}
        for r in possible_reshapes:
            keep_track[r] = {}
            # print(f'Try reshape r = {r}\n')
            force_Y_reshaped, λ_reshaped = self.reshape(self.λ, self.force_Y, r)
            new_guess_λ_0 = int(λ_reshaped.size/4) # random guess
            popt_pre, popt_post = self.fit(λ_reshaped, force_Y_reshaped, new_guess_λ_0)
            x = np.concatenate((np.linspace(min(λ_reshaped), λ_reshaped[new_guess_λ_0], new_guess_λ_0), 
                                np.linspace(λ_reshaped[new_guess_λ_0], max(λ_reshaped), len(λ_reshaped)-new_guess_λ_0)))
            theor_fit = self._heaviside_fitting(x, new_guess_λ_0, *popt_pre, *popt_post)
            f_rupture = [theor_fit[new_guess_λ_0-1]]
            f_rupture_next = [theor_fit[new_guess_λ_0]]
            delta_F = f_rupture[0]-f_rupture_next[0]
            keep_track[r]['old_chi'] = np.inf 
            for iter in range(max_iter):
                diff = force_Y_reshaped - theor_fit
                outliers = np.logical_or(diff < np.mean(diff)-1.5*np.std(diff), diff > np.mean(diff)+1.5*np.std(diff))

                if len(outliers) != 0:
                    for new_guess_λ_0 in np.where(outliers == True)[0]:
                        if new_guess_λ_0 < 5 or new_guess_λ_0 >= len(diff)-5: 
                            continue
                        
                        popt_pre, popt_post = self.fit(λ_reshaped, force_Y_reshaped, new_guess_λ_0)
                        x = np.concatenate((np.linspace(min(λ_reshaped), λ_reshaped[new_guess_λ_0], new_guess_λ_0), 
                                        np.linspace(λ_reshaped[new_guess_λ_0], max(λ_reshaped), len(λ_reshaped)-new_guess_λ_0)))
                        
                        theor_fit = self._heaviside_fitting(x, new_guess_λ_0, *popt_pre, *popt_post)
                    
                        f_rupture = [theor_fit[new_guess_λ_0-1]]
                        f_rupture_next = [theor_fit[new_guess_λ_0]]
                        delta_F = f_rupture[0]-f_rupture_next[0]
                        
                        chi = self.chi_squared(force_Y_reshaped, theor_fit, new_guess_λ_0)
                        
                        if (chi <= keep_track[r]['old_chi']) and (0.1 < delta_F < 1.) and λ_reshaped[new_guess_λ_0] > 0.12:
                            keep_track[r]['old_chi'] = chi
                            keep_track[r]['best_iter'] = iter
                            guess_λ_0 = new_guess_λ_0
                            keep_track[r]['guess_λ_0'] = guess_λ_0
                            best_popt_pre, best_popt_post = popt_pre, popt_post
                            keep_track[r]['best_popt_pre'] = best_popt_pre
                            keep_track[r]['best_popt_post'] = best_popt_post

                else:
                    new_guess_λ_0 = np.where(diff == min(diff))[0][0]
                    # print('No outliers found')
                
            # if show_per_reshape:
            #     plt.plot(λ_reshaped, force_Y_reshaped)
            #     plt.plot(λ_reshaped, theor_fit)
            #     plt.vlines(λ_reshaped[guess_λ_0], min(force_Y), max(force_Y), color='k', ls=':')
            #     plt.title(f'Reshape = {r}')
            #     plt.show()

        wh = np.array([keep_track[r]['old_chi'] for r in possible_reshapes])
        best_reshape = possible_reshapes[np.where(wh == min(wh))[0][-1]]


        best_chi = keep_track[best_reshape]['old_chi']
        best_iter = keep_track[best_reshape]['best_iter']
        guess_λ_0 = keep_track[best_reshape]['guess_λ_0']
        best_popt_pre = keep_track[best_reshape]['best_popt_pre']
        best_popt_post = keep_track[best_reshape]['best_popt_post']

        # print(f'The minimum chi squared ({best_chi}) has been found at iteration {best_iter}')
        # print(f'Best reshape r = {best_reshape}; Best index: {guess_λ_0}')


        force_Y, λ = self.reshape(self.λ, self.force_Y, best_reshape)

        x = np.concatenate((np.linspace(min(λ), λ[guess_λ_0], guess_λ_0),
                    np.linspace(λ[guess_λ_0], max(λ), len(λ)-guess_λ_0)))
        theor_fit = self._heaviside_fitting(x, guess_λ_0, *best_popt_pre, *best_popt_post)

        self.f_rupture = [theor_fit[guess_λ_0-1]]
        self.f_rupture_next = [theor_fit[guess_λ_0]]
        delta_F = self.f_rupture[0] - self.f_rupture_next[0]
        # print(f'Method2: Delta F = {delta_F}')
        
        self.t_0, self.k_eff, self.x_ssDNA, self.N_nucleotides = self.reading(self.time, best_popt_pre, guess_λ_0, self.f_rupture, self.f_rupture_next)

        self.index = [guess_λ_0]
        self.λ_0 = λ[guess_λ_0]
        self.best_popt_pre = best_popt_pre.tolist()
        self.best_popt_post = best_popt_post.tolist()
        self.params = self.f_rupture + self.f_rupture_next + self.x_ssDNA + self.N_nucleotides + [self.k_eff] + self.t_0 + [self.λ_0] + self.best_popt_pre + self.best_popt_post + [2]
        self.theor_f = theor_fit
        self.best_reshape = best_reshape
        self.force_Y, self.λ = force_Y, λ
        self.best_chi = best_chi
        print(len(self.params))



    def minimize_chi(self, ty, r=10, fitting_points=15, save=''):
        """
        This method allows to calculate the rip in a new different way.
        Instead of performing a reshaped analysis, a default reshape equal
        r = 10 is performed. Then a fit is performed - point by point: 
        the rip corresponds to the min χ^2.  

        Args:
            ty (str): The type of file: 'u' means unfolding sequence, otherwise
                                        'f' stands for folding trajectory.
            r (int):    Number of reshape points. Default to 10
            fitting_points (int):   Number of fitting points used to compute the fit
                                    and estimate the χ^2. Defaults to 15, so 30 points
                                    used for fitting: 15 before and 15 after the rip.         
            save (str): Allows to save the graph in order to create then a gif. 
                                If so, one must specify the whole folder name where
                                the .png(s) will be saved. Default to False.
        """

        force_Y_reshaped, λ_reshaped = self.reshape(self.λ, self.force_Y, r)
        # The idea is to try a fit every top points and evaluate the chi^2
        fitting_points = fitting_points
        best_chi = np.inf
        all_chi = []
        guess_λ_0 = 0
        self.best_state = False
        draw = Draw('0nM', '10') # just a placeholder for the draw object
        if ty == 'u':
            vals = range(np.where(λ_reshaped < 0.70)[0][-1], np.where(λ_reshaped > 0.05)[0][0], -1)  
        else:
            vals = range(np.where(λ_reshaped > 0.05)[0][0], np.where(λ_reshaped < 0.7)[0][-1], 1)
        
        for i in vals:

            λ = λ_reshaped[i-fitting_points:i+fitting_points]
            f_Y = force_Y_reshaped[i-fitting_points:i+fitting_points]
            popt_pre, popt_post = self.fit(λ, f_Y, fitting_points)
            x = np.concatenate((np.linspace(min(λ_reshaped), λ_reshaped[i], i),
                                np.linspace(λ_reshaped[i], max(λ_reshaped), len(λ_reshaped)-i)))
            theor_fit = self._heaviside_fitting(x, i, *popt_pre, *popt_post)
            chi = self.chi_squared(f_Y, theor_fit[i-fitting_points:i+fitting_points], fitting_points)
            f_rupture = [theor_fit[i-1]]
            f_rupture_next = [theor_fit[i]]
            delta_F = f_rupture[0] - f_rupture_next[0]
            if save:
                draw.make_plot(λ_reshaped, force_Y_reshaped, theor_fit, 2, '1', [i], fitting_points, 
                           0, f_rupture, f_rupture_next, '', 
                           save_fig=True, folder=save, number=0, N=i, ty='u', txt=r'$\chi^2$'+f'={chi}')

            all_chi.append(chi)

            # and (0.1 < delta_F < 1.) 

            if chi < best_chi and (0.1 < delta_F < 1.): # and λ_reshaped[i] > 0.12:
                # print('Best state found!')
                best_chi = chi
                guess_λ_0 = i
                self.best_state = True
                best_popt_pre, best_popt_post = popt_pre, popt_post


        if self.best_state:
            # print('Best state found!')
            best_popt_pre, best_popt_post = self.fit(λ_reshaped, force_Y_reshaped, guess_λ_0)
            # best_popt_pre, best_popt_post = self.fit(λ_reshaped[guess_λ_0-fitting_points:guess_λ_0+fitting_points],
            #                                          force_Y_reshaped[guess_λ_0-fitting_points:guess_λ_0+fitting_points],
            #                                          fitting_points)
            
            x = np.concatenate((np.linspace(min(λ_reshaped), λ_reshaped[guess_λ_0], guess_λ_0),
                                np.linspace(λ_reshaped[guess_λ_0], max(λ_reshaped), len(λ_reshaped)-guess_λ_0)))
            theor_fit = self._heaviside_fitting(x, guess_λ_0, *best_popt_pre, *best_popt_post)

            f_rupture = [theor_fit[guess_λ_0-1]]
            f_rupture_next = [theor_fit[guess_λ_0]]
            delta_F = f_rupture[0] - f_rupture_next[0]

            t_0, k_eff, x_ssDNA, N_nucleotides = self.reading(self.time, 
                                                              best_popt_pre if ty == 'u' else best_popt_post, 
                                                              guess_λ_0, 
                                                              f_rupture, 
                                                              f_rupture_next)

            index = [guess_λ_0]
            λ_0 = λ_reshaped[guess_λ_0]
            best_popt_pre = best_popt_pre.tolist()
            best_popt_post = best_popt_post.tolist()
            params = f_rupture + f_rupture_next + x_ssDNA + N_nucleotides + [k_eff] + t_0 + [λ_0] + best_popt_pre + best_popt_post + [2]
            theor_f = theor_fit
            best_reshape = r
            best_chi = best_chi
            # print(f'Best chi = {best_chi} with fitting_points = {fitting_points}\nDelta F = {delta_F}')
            # self.force_Y, self.λ = force_Y_reshaped, λ_reshaped
            return {'f_rupture':f_rupture, 'f_rupture_next':f_rupture_next, 't_0':t_0, 'k_eff':k_eff, 'x_ssDNA':x_ssDNA, 
                    'N_nucleotides':N_nucleotides, 'index':index, 'λ_0':λ_0, 'best_popt_pre':best_popt_pre, 
                    'best_popt_post':best_popt_post, 'params':params, 'theor_f':theor_f, 'best_reshape':best_reshape, 
                    'best_chi':best_chi, 'force_Y_reshaped':force_Y_reshaped, 'λ_reshaped':λ_reshaped, 
                    'fitting_points':fitting_points, 'delta_F':delta_F, 'best_state':True, 'N_fits':2}
        else:
            return self.no_output()
        
    def no_output(self):
        keys = ['f_rupture', 'f_rupture_next', 't_0', 'k_eff', 'x_ssDNA', 'N_nucleotides', 'index',
                'λ_0', 'best_popt_pre', 'best_popt_post', 'params', 'theor_f', 'best_reshape', 'best_chi',
                'force_Y_reshaped', 'λ_reshaped', 'fitting_points', 'delta_F', 'best_state', 'N_fits']
        vals = np.zeros(shape=len(keys), dtype=int)
        d = dict(zip(keys, vals))
        d['force_Y_reshaped'] = self.force_Y
        d['λ_reshaped'] = self.λ
        d['params'], d['index'], d['λ_0'], d['f_rupture'], d['f_rupture_next'], d['x_ssDNA'], d['N_nucleotides'], d['t_0'], d['best_popt_pre'], d['best_popt_post'] = self._errors()
        return d

    def _errors(self):
        index, λ_0, f_rupture, f_rupture_next, x_ssDNA, N_nucleotides, k_eff = [[0]]*7
        t_0 = [0]
        popt_pre, popt_post = [0, 0], [0, 0]
        params = f_rupture + f_rupture_next + x_ssDNA + N_nucleotides + k_eff + t_0 + λ_0 + popt_pre + popt_post + [0]
        return params, index, λ_0, f_rupture, f_rupture_next, x_ssDNA, N_nucleotides, t_0, popt_pre, popt_post


    def find_rip(self, ty, save='', print_out=False):

        """
        Method introduced with version 4.24 (April 2024): the rip point is found by minimizing χ^2 among both
        possible reshapes and number of fitting points. Then instead of just selecting the lowest χ^2, the 
        `best_state` is the one leading to the largest ΔF.

        Args:
            ty (str): The type of file: 'u' means unfolding sequence, otherwise
                                        'f' stands for folding trajectory.
        """

        fitting_points = [5, 10, 15, 20, 25] #, 30, 35, 40, 45, 50]
        possible_reshapes = [1, 5, 10, 15, 20] # , 25, 30
        all_res = []
        for r in possible_reshapes:
            res = []
            for p in fitting_points:
                if print_out:
                    print(f'Trying reshape = {r}')
                    print(f'Trying fitting_points = {p}')
                try:
                    output = self.minimize_chi(ty=ty, r=r, fitting_points=p, save=save)
                    res.append(output)
                except:
                    if print_out: 
                        print(f'Unable to perform fit with r = {r}, p = {p}')
                    output = self.no_output()
                    res.append(output)

            all_res.append(res)

        """    
        Find the result with min χ^2:
        instead of looking at the minimum χ^2 used as method up to now,
        I look at the fit maximizing ΔF:
        """
        self.all_res = all_res
        values = np.array([np.array([d['delta_F'] for d in row]) for row in all_res])
        # values = np.array([np.array([d['best_chi'] if d['best_chi'] != 0 else np.Inf for d in row]) for row in all_res])
        min_index = np.unravel_index(np.argmax(values), values.shape)
        # min_index = np.unravel_index(np.argmin(values), values.shape)

        [self.f_rupture, self.f_rupture_next, self.t_0, self.k_eff, self.x_ssDNA, 
         self.N_nucleotides, self.index, self.λ_0, self.best_popt_pre, self.best_popt_post, 
         self.params, self.theor_f, self.best_reshape, self.best_chi, self.force_Y, 
         self.λ, self.fitting_points, self.delta_F, self.best_state, self.N_fits] = list(all_res[min_index[0]][min_index[1]].values()) 



    def _heaviside_fitting(self, x, λ_0, a, b, c, d):
        return np.concatenate((x[:λ_0]*a+b, x[λ_0:]*c+d))


    # Piecewise-linear function
    def piecewise_linear(self, λ, guess_λ_0, a1, b1, a2, b2):
        """Piecewise-linear fit with two segments:
        - a1*x + b1 per x < x0
        - a2*x + b2 per x >= x0
        """
        return np.piecewise(
            λ,
            [λ < guess_λ_0, λ >= guess_λ_0],
            [lambda x: a1*x + b1, lambda x: a2*x + b2]
        )

    def fit_piecewise_linear(self, λ, force_Y, guess_λ_0):
        initial_guess = [guess_λ_0, 1, 0, 1, 0]
        params, _ = curve_fit(self.piecewise_linear, λ, force_Y, p0=initial_guess)
        return params


    def fit(self, λ, force_Y, guess_λ_0):
        pre_λ = λ[:guess_λ_0]
        pre_f = force_Y[:guess_λ_0]
        popt_pre, _ = curve_fit(lambda x, a, b: x*a+b, pre_λ, pre_f)
        post_λ = λ[guess_λ_0:]
        post_f = force_Y[guess_λ_0:]
        popt_post, _ = curve_fit(lambda x, a, b: x*a+b, post_λ, post_f)
        return popt_pre, popt_post
    
    # The idea is to minimize a chi_squared function or whatever
    def chi_squared(self, data, theor, guess, fitting_params = 5):
        points = min(int(len(data)/4), 15)
        # chi = np.sum(((data[guess-points:guess+points]-theor[guess-points:guess+points])/np.std(data[guess-points:guess+points]))**2)
        chi = np.sum(((data-theor)/np.std(data))**2)/fitting_params
        # /(len(data[guess-points:guess+points]) - fitting_params)
        return round(chi, 3)


    # Reshape function
    def reshape(self, λ, force_Y, n_points):
        num_segments = len(force_Y) // n_points

        # Reshape the data into segments of n_points points
        force_Y_reshaped = force_Y[:num_segments*n_points].reshape(num_segments, n_points)
        λ_reshaped = λ[:num_segments*n_points].reshape(num_segments, n_points)
        
        # Calculate the median of each segment: median filter
        force_Y_reshaped = np.median(force_Y_reshaped, axis=1)
        λ_reshaped = np.median(λ_reshaped, axis=1)

        return force_Y_reshaped, λ_reshaped

