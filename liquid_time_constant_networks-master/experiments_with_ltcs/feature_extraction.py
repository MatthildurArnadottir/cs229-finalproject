import pandas as pd

from interest_rate import load_crappy_formated_csv_ir
from energy import load_crappy_formated_csv_energy
import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

def interest_rate_pca():
    """
    Interest rate data
    """
    X, y, dates = load_crappy_formated_csv_ir("data/interest_rate_data/interest_rate_data_all.txt", number_of_cols=7)
    features = ['CPI','GDP','S&P500','Treasury Bond - 1 year','Treasury Bond - 10 year','Unemployment','Debt']

    # Visualize original dataset features over time
    for j in range(X.shape[1]):
        plt.plot(X[:,j], label = features[j])
    plt.legend()
    plt.xlabel('Time')
    plt.xticks([], [])
    plt.title("Interest rate dataset")
    plt.show()

    X_df = pd.DataFrame(X)
    X_df.columns = features
    pca = PCA(n_components = 3)
    X_pca = pca.fit_transform(X)
    X_pca_df = pd.DataFrame(X_pca)
    X_pca_df.insert(0, 'Label', y)
    np.savetxt(r"data/interest_rate_data/pca_data_3.txt", X_pca_df, fmt="%s", delimiter=';')
    ratios = pca.explained_variance_ratio_
    print("Explained variance per component: ")
    print(ratios)

    # 3D plot of principal components
    ax = plt.axes(projection="3d")
    ax.scatter3D(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c = y)
    ax.set_xlabel('PCA component 1')
    ax.set_ylabel('PCA component 2')
    ax.set_zlabel('PCA component 3')
    ax.set_title("Interest rate dataset PCA")
    plt.show()

def energy_consumption_pca():
    """
    Energy consumption data
    """
    X, y = load_crappy_formated_csv_energy()
    X_df = pd.DataFrame(X)
    pca = PCA(n_components = 7)
    X_pca = pca.fit_transform(X_df)
    X_pca_df = pd.DataFrame(X_pca)
    X_pca_df.insert(0, 'Label', y)
    np.savetxt(r"data/energy_data/pca_data_7.txt", X_pca_df, fmt="%s", delimiter=',')
    ratios = pca.explained_variance_ratio_
    print("Explained variance per component: ")
    print(ratios)

    ax = plt.axes(projection="3d")
    ax.scatter3D(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c = y)
    ax.set_xlabel('PCA component 1')
    ax.set_ylabel('PCA component 2')
    ax.set_zlabel('PCA component 3')
    ax.set_title("Energy consumption dataset PCA")
    plt.show()

if __name__ == "__main__":
    print("Doing PCA for interest rate dataset")
    interest_rate_pca()

    print("Doing PCA for energy consumption dataset")
    energy_consumption_pca()






