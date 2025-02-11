class AviationVisualizer:
    def __init__(self, df=None):
        self.df = df
    
    def update_dataframe(self, df):
        self.df = df 