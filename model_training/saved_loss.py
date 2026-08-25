import pandas as pd

class loss_out:
    def __init__(self,path):
        self.loss_val_noise = []
        self.loss_train = []
        self.loss_val_clean = []
        self.acc_val_noise = []
        self.acc_train = []
        self.acc_val_clean = []
        self.path = path
    
    def update_loss_val_clean(self,data):
        self.loss_val_clean.append(data)

    def update_loss_train(self,data):
        self.loss_train.append(data)

    def update_acc_val_clean(self,data):
        self.acc_val_clean.append(data)

    def update_acc_train(self,data):
        self.acc_train.append(data)

    def update_loss_val_noise(self,data):
        self.loss_val_noise.append(data)

    def update_acc_val_noise(self,data):
        self.acc_val_noise.append(data)

    def save_csv(self):
        
        df = pd.DataFrame({"loss_val_noise" : self.loss_val_noise,
                            "loss_train" : self.loss_train,
                            "acc_val_noise" : self.acc_val_noise,
                            "acc_train" : self.acc_train,
                            "loss_val_clean": self.loss_val_clean,
                            "acc_val_clean" : self.acc_val_clean})

        df.to_csv(self.path, index=True)
        
        return