import os
import numpy as np
import string

# import pandas as pd
# import matplotlib.pyplot as plt

class Txt_Reading:
    def __init__(self, folder, f_MAX) -> None:
        self.folder = folder
        # self.dir_name = 'G:/Il mio Drive/Fisica/Dispense Università/2 anno/Zaltron & Xavi/' + dir_name
        self.dir_name = 'F:/Zaltron & Xavi/' + self.folder + '/'
        self.f_MAX = str(f_MAX)
        self.name = f'pull{self.f_MAX}'

        self.working_dir = f'Data/{self.folder}/{self.f_MAX}'
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            self._createTxt()
        

    
    def readTxt(self, number = 1, N = 1, ty = 'u'):
        self.number = number   # molecule number
        self.path = f'{self.working_dir}/{self.number}/{N}_{ty}.txt'

        self.file = np.loadtxt(self.path)
        
        return self.file


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