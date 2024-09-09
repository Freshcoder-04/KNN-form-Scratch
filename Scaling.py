class UnitNormalScaling:
    def __init__(self,df):
        self.df = df
        self.columns = df.columns
    def Scale(self):
        for x in self.columns:
            if(self.df[x].dtype == 'float64'):
                stdDev = self.df[x].std()
                Mean = self.df[x].mean()
                self.df[x] = (self.df[x]-Mean)/stdDev

class MinMaxScaling:
    def __init__(self,df):
        self.df = df
        self.columns = df.columns
    def Scale(self):
        for x in self.columns:
            if(self.df[x].dtype == 'float64'):
                max = self.df[x].max()
                min = self.df[x].min()
                self.df[x] = (self.df[x] - min)/(max - min)