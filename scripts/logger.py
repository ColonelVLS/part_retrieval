import matplotlib.pyplot as plt
from IPython.display import display, clear_output

class LivePlot():

    def __init__(self, specs):
        
        '''
            Expecting a dictionary like so:
            {
                "on_step": [
                    {"graph1_title" : "y_label1"},
                    {"graph2_title" : "y_label2"}
                ],

                "on_epoch": [
                    {"graph3_title" : "y_label3"},
                    {"graph4_title" : "y_label4"}
                ],

                "on_update": [
                    {"graph5_title" : "y_label5"},
                    {"graph6_title" : "y_label6"}
                ]
            }

            The graphs will be made dynamically and split in rows of 3 accordingly.
            The values will be updated through the functions on_step and on_epoch.
            Each function will ask the user for as many values as there are keys in the dictionary.
        '''  

        self.specs = specs

        self.on_step_titles = [list(d.keys())[0] for d in specs["on_step"]]
        self.on_step_ylabels = [list(d.values())[0] for d in specs["on_step"]]
        
        self.on_epoch_titles = [list(d.keys())[0] for d in specs["on_epoch"]]
        self.on_epoch_ylabels = [list(d.values())[0] for d in specs["on_epoch"]]
        
        self.on_update_titles = [list(d.keys())[0] for d in specs["on_update"]]
        self.on_update_ylabels = [list(d.values())[0] for d in specs["on_update"]]

        self.on_step_samples = [[] for _ in range(len(self.on_step_titles))]
        self.on_epoch_samples = [[] for _ in range(len(self.on_epoch_titles))]
        self.on_update_samples = [[] for _ in range(len(self.on_update_titles))]

        self.current_step = 0
        self.current_epoch = 0

        self.ns = len(self.on_step_titles)
        self.ne = len(self.on_epoch_titles)
        self.nu = len(self.on_update_titles)
        total_plots = self.ns + self.ne + self.nu
        self.org = f"{1 + total_plots // 3}{3}"

    def update(self):
        
        clear_output(wait=True)

        plt.figure(figsize=(12, 4))

        #drawing on step plots
        for i, (title, ylabel) in enumerate(zip(self.on_step_titles, self.on_step_ylabels)):

            plt.subplot(int(self.org + str(i+1)))
            plt.plot(self.on_step_samples[i], label=ylabel)
            plt.xlabel("Step")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()

        #drawing on epoch plots
        for i, (title, ylabel) in enumerate(zip(self.on_epoch_titles, self.on_epoch_ylabels)):

            plt.subplot(int(self.org + str(i+1+self.ns)))
            plt.plot(self.on_epoch_samples[i], label=ylabel)
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()

        #drawing on custom update plots
        for i, (title, ylabel) in enumerate(zip(self.on_update_titles, self.on_update_ylabels)):

            plt.subplot(int(self.org + str(i + 1 + self.ns + self.ne)))
            plt.plot(self.on_update_samples[i], label=ylabel)
            plt.xlabel("k")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()

        plt.tight_layout()
        plt.show()
    
    def on_epoch(self, samples):
        
        if len(samples) != self.ne:
            raise ValueError("The number of samples provided does not match the number of plots.")
        
        for i, sample in enumerate(samples):
            self.on_epoch_samples[i].append(sample)

        #emptying the on_sample lists
        for i in range(self.ns):
            self.on_step_samples[i] = []

        self.update()

    def on_step(self, samples):

        if len(samples) != self.ns:
            raise ValueError("The number of samples provided does not match the number of plots.")
        
        for i, sample in enumerate(samples):
            self.on_step_samples[i].append(sample)

        self.update()

    def on_custom_update(self, samples):

        if len(samples) != self.nu:
            raise ValueError("The number of samples provided does not match the number of plots.")

        #on custom update does not expect a single sample, but rather the entire list of samples
        #therefore we do not simply add to the list, we replace it with the new one        
        for i, sample in enumerate(samples):
            self.on_update_samples[i] = sample

        self.update()

    def load_state_dict(self, sd):

        self.specs = sd["specs"]
        self.on_step_samples = sd["on_step_samples"]
        self.on_epoch_samples = sd["on_epoch_samples"]
        self.on_update_samples = sd["on_update_samples"]

        self.on_step_titles = [list(d.keys())[0] for d in self.specs["on_step"]]
        self.on_step_ylabels = [list(d.values())[0] for d in self.specs["on_step"]]
        
        self.on_epoch_titles = [list(d.keys())[0] for d in self.specs["on_epoch"]]
        self.on_epoch_ylabels = [list(d.values())[0] for d in self.specs["on_epoch"]]
        
        self.on_update_titles = [list(d.keys())[0] for d in self.specs["on_update"]]
        self.on_update_ylabels = [list(d.values())[0] for d in self.specs["on_update"]]

        self.ns = len(self.on_step_titles)
        self.ne = len(self.on_epoch_titles)
        self.nu = len(self.on_update_titles)
        total_plots = self.ns + self.ne + self.nu
        self.org = f"{1 + total_plots // 3}{3}"

    def state_dict(self):

        return {
            "specs": self.specs,
            "on_step_samples": self.on_step_samples,
            "on_epoch_samples": self.on_epoch_samples,
            "on_update_samples": self.on_update_samples
        }