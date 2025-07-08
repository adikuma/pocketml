import pandas as pd

class OrdinalEncoder:
    def __init__(self):
        self.maps = {}

    def fit(self, df, columns):
        for col in columns:
            cats = df[col].unique()
            self.maps[col] = {v: i for i, v in enumerate(cats)}
        return self

    def transform(self, df):
        df = df.copy()
        for col, m in self.maps.items():
            df[col] = df[col].map(m)
        return df

    def fit_transform(self, df, columns):
        return self.fit(df, columns).transform(df)