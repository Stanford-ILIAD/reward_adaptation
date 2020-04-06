import pandas as pd

def get_weights_from_csv(csv_file):
    """
    Extracts weights and returns in form of scatter plot input
    """
    df = pd.read_csv(csv_file)
    df = df.drop(['epoch', 'std'], axis=1)
    x = df['weightx'].values
    y = df['weighty'].values
    z = df['loss_pi'].values
    return x, y, z
