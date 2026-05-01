import pandas as pd

class ExcelTransformation():
    def __init__(self, df: pd.DataFrame, steps: str) -> pd.DataFrame:
        self.df = df
        self.steps = steps.split(',')
        self.res_df = self._perform_execution()

    def _view_data(self):
        print(self.df)
        print(self.steps)

    def _perform_execution(self):
        for step in self.steps:
            # exec(step, global_variable, local_variable)
            exec(step.strip(), {'pd': pd}, {'df': self.df})
        return self.df

if __name__ == '__main__':
    df = pd.DataFrame({
        'a': [1,2,3,4,5],
        'b': [1,2,3,4,5]
    })
    steps = "df['res'] = df['a'] + df['b'], df['mul_res'] = df['a'] * df['b'], df['even_or_odd'] = df['a'].apply(lambda x: 'Even' if x % 2 == 0 else 'Odd')"
    excl_trnsf = ExcelTransformation(df=df, steps=steps)
    # excl_trnsf._view_data()
    # excl_trnsf._perform_execution()
    print(excl_trnsf.df)
