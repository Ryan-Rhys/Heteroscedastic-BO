# Author: Ryan-Rhys Griffiths
"""
Utility functions for parsing chemical datasets.
"""

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def parse_dataset(task_name, path, use_fragments=True, use_exp=True):
    """
    Returns list of molecular smiles, as well as the y-targets of the dataset
    :param task_name: name of the task
    :param path: dataset path
    :param use_fragments: If True return fragments instead of SMILES
    :return: x, y where x can be SMILES or fragments and y is the label.
    """

    if task_name == 'FreeSolv':
        df = pd.read_table(path, delimiter=';')
        smiles_list = df[' SMILES'].tolist()
        exp = df[' experimental value (kcal/mol)'].to_numpy()  # can change to df['calc'] for calculated values
        exp_std = df[' experimental uncertainty (kcal/mol)'].to_numpy()
        calc = df[' Mobley group calculated value (GAFF) (kcal/mol)'].to_numpy()
        calc_std = df[' calculated uncertainty (kcal/mol)'].to_numpy()
    else:
        raise Exception('Must provide task name')

    if use_fragments:

        # descList[115:] contains fragment-based features only (https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html)

        fragments = {d[0]: d[1] for d in Descriptors.descList[115:]}
        x = np.zeros((len(smiles_list), len(fragments)))
        for i in range(len(smiles_list)):
            mol = MolFromSmiles(smiles_list[i])
            try:
                features = [fragments[d](mol) for d in fragments]
            except:
                raise Exception('molecule {}'.format(i) + ' is not canonicalised')
            x[i, :] = features

    else:  # Use 512-bit Morgan fingerprints

        rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles_list]
        x = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512) for mol in rdkit_mols]
        x = np.asarray(x)

    if use_exp:
        y = exp
        std = exp_std
    else:
        y = calc
        std = calc_std

    return x, y, std


def transform_data(X_train, X_test, y_train, y_test, n_components):
    """
    Apply feature scaling, dimensionality reduction to the data. Return the standardised and low-dimensional train and
    test sets together with the scaler object for the target values.

    :param X_train: input train data
    :param X_test: input test data
    :param y_train: train labels
    :param y_test: test labels
    :param n_components: number of principal components to keep
    :return: X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler, x_scaler, pca_transform
    """

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
    pca = PCA(n_components)
    X_train_scaled = pca.fit_transform(X_train_scaled)
    print('Fraction of variance retained is: ' + str(sum(pca.explained_variance_ratio_)))
    X_test_scaled = pca.transform(X_test_scaled)
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler
