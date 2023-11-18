import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button


class Draw:
    def __init__(self, folder, f_MAX) -> None:
        self.folder = folder
        # self.dir_name = 'G:/Il mio Drive/Fisica/Dispense Università/2 anno/Zaltron & Xavi/' + dir_name
        self.f_MAX = str(f_MAX)
        self.name = f'pull{self.f_MAX}'
        
        self.colors = ["#7afdd6","#77ff94","#a1e44d","#60993e","#613a3a", 
                       "#e3b505","#95190c","#610345","#107e7d","#044b7f", 
                       "#20bf55","#0b4f6c","#01baef","#fbfbff","#757575"]
        self.markers = ["D", "o", "v", "^", "1", "8", "s", "p", "*", "x", 
                        "d", "|", "_", "4", "P"]

        # Create folder for images
        self.working_dir = f'imgs/{self.folder}/{self.f_MAX}/'
        isExist = os.path.exists(self.working_dir)
        if not isExist:
            os.makedirs(self.working_dir)
            print("The images directory is created...")

            
    def draw_fit_diff(self, λ_filtered, diff, std, theor_std, points, number, N, save=True):
        # fig = plt.figure()
        plt.plot(λ_filtered, diff)
        plt.scatter(points[0], points[1], marker='D', color='black', linewidths=2.)
        plt.hlines(std, min(λ_filtered), max(λ_filtered), ls='--', color='red')
        plt.hlines(-std, min(λ_filtered), max(λ_filtered), ls='--', color='red')
        plt.hlines(theor_std, min(λ_filtered), max(λ_filtered), ls='--', color='green')
        plt.hlines(-theor_std, min(λ_filtered), max(λ_filtered), ls='--', color='green')
        plt.title(f'{number}_{N}_{self.ty}')
        # if save:
        #     plt.savefig(f'{self.working_dir}/{number}_{N}_{self.ty}.png', dpi=100)
        #     plt.close(fig)
        # else:
        plt.show()


    def onselect(self, eclick, erelease):

        # Callback function for RectangleSelector
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print(f"Selected region: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        # Update the plot based on the selected region
        self.ax.set_xlim(min(x1, x2), max(x1, x2))
        self.ax.set_ylim(min(y1, y2), max(y1, y2))

        # Retrieve and store the selected data points
        mask = (self.λ >= min(x1, x2)) & (self.λ <= max(x1, x2)) & (self.force >= min(y1, y2)) & (self.force <= max(y1, y2))
        # self.λ = np.array(self.λ)[mask]
        # self.force = np.array(self.force )[mask]

        plt.draw()
        # self.output = mask

    def reshape_callback(self, event, path):
        self.ax.clear()
        self.n_points += 5
        num_segments = len(self.force) // self.n_points

        # Reshape the data into segments of n_points points
        force_Y_reshaped = self.force[:num_segments*self.n_points].reshape(num_segments, self.n_points)
        λ_reshaped = self.λ[:num_segments*self.n_points].reshape(num_segments, self.n_points)
        
        # Calculate the median of each segment: median filter
        force_Y_reshaped = np.median(force_Y_reshaped, axis=1)
        λ_reshaped = np.median(λ_reshaped, axis=1)

        self.ax.plot(λ_reshaped, force_Y_reshaped)
        self.ax.set_title(f'{path}')
        plt.draw()


    def reset_callback(self, event, path):
        self.n_points = 0
        self.ax.clear()
        self.ax.plot(self.λ, self.force)
        self.ax.set_title(f'{path}')
        plt.draw()

    def jump_callback(self, event, jumps):
        plt.close()
        self.output = 'jump'

    def linear_callback(self, event, linears):
        plt.close()
        self.output = 'linear'

    def trash_callback(self, event, trashes):
        plt.close()
        self.output = 'trash'

    def unknown_callback(self, event, unknown):
        plt.close()
        self.output = 'unknown'

    def stop_callback(self, event, finish):
        plt.close()
        self.output = 'stop'
        
    

    def identify_by_hand(self, λ, force, path, jumps, linears, trashes, unknown, finish):
        self.output = '0'
        self.λ, self.force = λ, force
        # Function to create the plot with zoom and linear buttons
        fig, self.ax = plt.subplots(figsize=(10,6))
        self.ax.plot(self.λ, self.force)
        self.ax.set_title(f'{path}')

        # Create RectangleSelector
        self.rs = RectangleSelector(self.ax, self.onselect, drawtype='box', 
                                    useblit=False, button=[1],
                                    minspanx=5, minspany=5, 
                                    spancoords='pixels', interactive=True)


        # Create Reshape button
        self.n_points = 0
        self.reshape_button = Button(plt.axes([0.75, 0.28, 0.1, 0.07]), 'Reshape +5')
        self.reshape_button.on_clicked(lambda event: self.reshape_callback(event, path))

        # Create Reset button
        self.reset_button = Button(plt.axes([0.85, 0.28, 0.1, 0.07]), 'Reset')
        self.reset_button.on_clicked(lambda event: self.reset_callback(event, path))

        # Create Jump button
        self.jump_button = Button(plt.axes([0.7, 0.2, 0.1, 0.07]), 'Jump')
        self.jump_button.on_clicked(lambda event: self.jump_callback(event, jumps))

        # Create Linear button
        self.linear_button = Button(plt.axes([0.8, 0.2, 0.1, 0.07]), 'Linear')
        self.linear_button.on_clicked(lambda event: self.linear_callback(event, linears))

        # Create Trash button
        self.trash_button = Button(plt.axes([0.9, 0.2, 0.1, 0.07]), 'Trash')
        self.trash_button.on_clicked(lambda event: self.trash_callback(event, trashes))     

        # Create Unknwon(?) button
        self.unknown = Button(plt.axes([0.8, 0.35, 0.1, 0.07]), '?', color='gray')
        self.unknown.on_clicked(lambda event: self.unknown_callback(event, trashes))        
   

        # Create Stop button
        self.stop_button = Button(plt.axes([0.8, 0.12, 0.1, 0.07]), 'Stop', color='red')
        self.stop_button.on_clicked(lambda event: self.stop_callback(event, finish))


        plt.show()


        if self.output == 'jump':
            jumps.append(path)
        elif self.output == 'linear':
            linears.append(path)
        elif self.output == 'trash':
            trashes.append(path)
        elif self.output == 'unknown':
            unknown.append(path)
        elif self.output == 'stop':
            finish = False


        return jumps, linears, trashes, unknown, finish
    

    def final_plots(self, molecules, all_molecules_f, all_molecules_u, show=True):
        # Create images
        fig, ax = plt.subplots(3, figsize=(12, 20))
        g = 0
        for m in range(len(molecules)):
            col = self.colors[m]
            for i in range(len(all_molecules_f[m])):
                g += 1
                lab = f'Molecule {molecules[m]}' if i == 0 else ''
                f, f_next, x_ssDNA, N_nucleotides, t_0 = all_molecules_f[m][i][0]
                # if f != 0:
                ax[0].scatter(t_0, f_next, c=col, marker=self.markers[m], label=lab)
                ax[1].scatter(g, x_ssDNA, c=col, marker=self.markers[m], label=lab)
                ax[2].scatter(g, N_nucleotides, c=col, marker=self.markers[m], label=lab)

        ax[0].grid()
        ax[0].legend(loc='best')
        ax[0].set_title(f'f_MAX = {self.f_MAX} -- Folding')
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('$f_{rupture} \\:[pN]$')

        ax[1].grid()
        ax[1].legend()
        ax[1].set_title('$x_{ssDNA}$')
        ax[1].set_ylabel('$x_{ssDNA}$')

        ax[2].grid()
        ax[2].legend()
        ax[2].set_title('# of nucleotides')
        ax[2].set_ylabel('$N_{nucleotides}$')

        plt.savefig(f'imgs/{self.folder}/{self.f_MAX}/f_MAX_{self.f_MAX}_Folding.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()

        fig, ax = plt.subplots(3, figsize=(12, 20))
        g = 0

        for m in range(len(molecules)):
            col = self.colors[m]
            for i in range(len(all_molecules_u[m])):
                g += 1
                lab = f'Molecule {molecules[m]}' if i == 0 else ''
                f, f_next, x_ssDNA, N_nucleotides, t_0 = all_molecules_u[m][i][0]
                # if f != 0:
                ax[0].scatter(t_0, f_next, c=col, marker=self.markers[m], label=lab)
                ax[1].scatter(g, x_ssDNA, c=col, marker=self.markers[m], label=lab)
                ax[2].scatter(g, N_nucleotides, c=col, marker=self.markers[m], label=lab)

        ax[0].grid()
        ax[0].legend(loc='best')
        ax[0].set_title(f'f_MAX = {self.f_MAX} -- Unfolding')
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('$f_{rupture} \\:[pN]$')

        ax[1].grid()
        ax[1].legend()
        ax[1].set_title('$x_{ssDNA}$')
        ax[1].set_ylabel('$x_{ssDNA}$')

        ax[2].grid()
        ax[2].legend()
        ax[2].set_title('# of nucleotides')
        ax[2].set_ylabel('$N_{nucleotides}$')

        plt.savefig(f'imgs/{self.folder}/{self.f_MAX}/f_MAX_{self.f_MAX}_Unfolding.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()


    def plots_per_single_molecule(self, molecules, all_molecules_f, all_molecules_u, show=False):
        # Plots per single molecule
        for m in range(len(molecules)):
            fig, ax = plt.subplots(3, figsize=(8, 18))
            col = self.colors[m]
            g = 0
            for i in range(len(all_molecules_f[m])):
                g += 1 
                lab = f'Molecule {molecules[m]}' if i == 0 else ''
                f, f_next, x_ssDNA, N_nucleotides, t_0 = all_molecules_f[m][i][0]
                ax[0].scatter(t_0, f, c=col, marker=self.markers[m], label=lab)
                ax[1].scatter(g, x_ssDNA, c=col, marker=self.markers[m], label=lab)
                ax[2].scatter(g, N_nucleotides, c=col, marker=self.markers[m], label=lab)
            ax[0].grid()
            ax[0].legend(loc='best')
            ax[0].set_title(f'f_MAX = {self.f_MAX} -- Folding')
            ax[0].set_xlabel('Time [s]')
            ax[0].set_ylabel('$f_{rupture} \\:[pN]$')

            ax[1].grid()
            ax[1].legend()
            ax[1].set_title('$x_{ssDNA}$')
            ax[1].set_ylabel('$x_{ssDNA}$')

            ax[2].grid()
            ax[2].legend()
            ax[2].set_title('# of nucleotides')
            ax[2].set_ylabel('$N_{nucleotides}$')
            plt.savefig(f'imgs/{self.folder}/{self.f_MAX}/f_MAX_{self.f_MAX}_Folding_Molecule{molecules[m]}.png', dpi=300, bbox_inches='tight')
            if show:
                plt.show()

        for m in range(len(molecules)):
            fig, ax = plt.subplots(3, figsize=(8, 18))
            col = self.colors[m]
            g = 0
            for i in range(len(all_molecules_u[m])):
                g += 1 
                lab = f'Molecule {molecules[m]}' if i == 0 else ''
                f, f_next, x_ssDNA, N_nucleotides, t_0 = all_molecules_u[m][i][0]
                ax[0].scatter(t_0, f, c=col, marker=self.markers[m], label=lab)
                ax[1].scatter(g, x_ssDNA, c=col, marker=self.markers[m], label=lab)
                ax[2].scatter(g, N_nucleotides, c=col, marker=self.markers[m], label=lab)
            ax[0].grid()
            ax[0].legend(loc='best')
            ax[0].set_title(f'f_MAX = {self.f_MAX} -- Unfolding')
            ax[0].set_xlabel('Time [s]')
            ax[0].set_ylabel('$f_{rupture} \\:[pN]$')

            ax[1].grid()
            ax[1].legend()
            ax[1].set_title('$x_{ssDNA}$')
            ax[1].set_ylabel('$x_{ssDNA}$')

            ax[2].grid()
            ax[2].legend()
            ax[2].set_title('# of nucleotides')
            ax[2].set_ylabel('$N_{nucleotides}$')
            plt.savefig(f'imgs/{self.folder}/{self.f_MAX}/f_MAX_{self.f_MAX}_Unfolding_Molecule{molecules[m]}.png', dpi=300, bbox_inches='tight')
            if show:
                plt.show()
