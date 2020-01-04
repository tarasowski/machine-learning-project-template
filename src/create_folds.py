import pandas as pd
from sklearn import model_selection
import os


if __name__ == '__main__':
    dir_input = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'input', 'train.csv'))
    dir_output = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'input', 'train_folds.csv'))
    df = pd.read_csv(dir_input)
    df['kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=42)


    for fold, (train_idx, val_idx) in enumerate(kf.split(X = df, y=df.target.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold

    df.to_csv(dir_output, index=False) 
