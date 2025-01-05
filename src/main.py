

import pandas as pd
import numpy as np
import math
from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error


#  1. Funcții ajutătoare (entropie, information gain)


def entropy(labels):
    """
    Calculează entropia pentru un set de etichete discrete.
    """
    counter = defaultdict(int)
    for lbl in labels:
        counter[lbl] += 1
    total = len(labels)
    e = 0.0
    for val in counter.values():
        p = val / total
        e -= p * math.log2(p)  # Entropia = -Σ(p_i * log2(p_i))
    return e

def information_gain(parent_labels, subsets):
    """
    Calculează gain-ul de informație după un split.
    parent_labels: toate etichetele din nodul părinte
    subsets: listă de liste, fiecare conținând un subset de etichete.
    """
    H_parent = entropy(parent_labels)
    n = len(parent_labels)
    H_children = 0.0
    for sub in subsets:
        H_children += (len(sub) / n) * entropy(sub)
    return H_parent - H_children


#  2. Clasa ID3Classifier (arbore de decizie pe date discrete)


class ID3Classifier:
    """
    Construiește un arbore de decizie ID3 și produce etichete discrete.
    """
    def __init__(self):
        self.tree_ = None
        self.features_ = None

    def fit(self, X, y, feature_names):
        """
        Antrenează modelul ID3 pe X (discret) și y (discret).
        """
        self.features_ = feature_names
        data = pd.DataFrame(X, columns=feature_names)
        data['__label__'] = y
        self.tree_ = self._build_tree(data, feature_names)

    def _build_tree(self, data, feature_names):
        """
        Construiește recursiv arborele. Returnează fie un nod-frunză,
        fie un nod cu subramuri.
        """
        labels = data['__label__'].values
        # Dacă toate valorile din labels sunt identice, se creează o frunză
        if len(set(labels)) == 1:
            return ('LEAF', labels[0])

        # Dacă nu mai există feature-uri pentru split, returnează eticheta majoritară
        if len(feature_names) == 0:
            return ('LEAF', self._majority_label(labels))

        best_ig = -1
        best_feat = None
        best_splits = {}

        for feat in feature_names:
            subsets = {}
            for val in data[feat].unique():
                subset_data = data[data[feat] == val]['__label__'].values
                subsets[val] = subset_data
            ig_feat = information_gain(labels, subsets.values())
            if ig_feat > best_ig:
                best_ig = ig_feat
                best_feat = feat
                best_splits = subsets

        if best_feat is None:
            return ('LEAF', self._majority_label(labels))

        # Creează un nod cu subramuri pentru fiecare valoare a feature-ului
        remaining_feats = [f for f in feature_names if f != best_feat]
        node = ('NODE', best_feat, {})

        for val, subset_labels in best_splits.items():
            subset_data = data[data[best_feat] == val].copy()
            if len(subset_data) == 0:
                node[2][val] = ('LEAF', self._majority_label(labels))
            else:
                node[2][val] = self._build_tree(subset_data, remaining_feats)

        return node

    def _majority_label(self, labels):
        """
        Alege eticheta majoritară dintr-un set de etichete.
        """
        counter = defaultdict(int)
        for lbl in labels:
            counter[lbl] += 1
        return max(counter, key=counter.get)

    def predict(self, X):
        """
        Produce etichete discrete pentru fiecare rând din X.
        """
        predictions = []
        for row in X:
            pred = self._predict_one(row, self.tree_, self.features_)
            predictions.append(pred)
        return np.array(predictions)

    def _predict_one(self, row, node, feature_names):
        """
        Parcurge recursiv arborele pentru un singur eșantion.
        """
        nod_type = node[0]
        if nod_type == 'LEAF':
            return node[1]
        elif nod_type == 'NODE':
            feat_name = node[1]
            branches = node[2]
            feat_idx = feature_names.index(feat_name)
            val = row[feat_idx]
            if val in branches:
                return self._predict_one(row, branches[val], feature_names)
            else:
                # Fallback, dacă apare o valoare neobservată
                leaves_labels = []
                for subnode in branches.values():
                    leaves_labels += self._get_all_leaves(subnode)
                return self._majority_label(leaves_labels)
        else:
            return 'UNKNOWN'

    def _get_all_leaves(self, node):
        """
        Extrage recursiv toate etichetele frunză din subarbore.
        """
        if node[0] == 'LEAF':
            return [node[1]]
        elif node[0] == 'NODE':
            results = []
            for subnode in node[2].values():
                results += self._get_all_leaves(subnode)
            return results
        return []


#  3. Clasa NaiveBayesDiscrete (date discrete)


class NaiveBayesDiscrete:
    """
    Implementează Naive Bayes pentru feature-uri discrete și
    o etichetă discretă (ex. sold discret).
    """
    def __init__(self):
        self.classes_ = None
        self.prior_ = {}
        self.likelihoods_ = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.counts_class_ = {}
        self.feature_names_ = None

    def fit(self, X, y, feature_names):
        """
        Antrenează Naive Bayes folosind X și y (discrete).
        """
        self.feature_names_ = feature_names
        self.classes_ = np.unique(y)

        for cl in self.classes_:
            self.counts_class_[cl] = 0

        # Numără aparițiile (P(C) și P(x|C))
        for i in range(len(y)):
            cl = y[i]
            self.counts_class_[cl] += 1
            for feat_idx, feat_val in enumerate(X[i]):
                self.likelihoods_[feat_idx][feat_val][cl] += 1

        total_samples = len(y)
        for cl in self.classes_:
            self.prior_[cl] = self.counts_class_[cl] / total_samples

        # Transformă numărătorile în probabilități
        for feat_idx in self.likelihoods_:
            for feat_val in self.likelihoods_[feat_idx]:
                for cl in self.likelihoods_[feat_idx][feat_val]:
                    self.likelihoods_[feat_idx][feat_val][cl] /= self.counts_class_[cl]

    def predict(self, X):
        """
        Produce etichete discrete pentru fiecare rând din X.
        """
        predictions = []
        for row in X:
            best_class = None
            max_prob = -1
            for cl in self.classes_:
                # Începe cu prior P(C)
                prob = self.prior_[cl]
                # Înmulțește cu P(xi|C) pentru fiecare feature
                for feat_idx, feat_val in enumerate(row):
                    if feat_val in self.likelihoods_[feat_idx]:
                        if cl in self.likelihoods_[feat_idx][feat_val]:
                            prob *= self.likelihoods_[feat_idx][feat_val][cl]
                        else:
                            prob *= 1e-6
                    else:
                        prob *= 1e-6
                if prob > max_prob:
                    max_prob = prob
                    best_class = cl
            predictions.append(best_class)
        return np.array(predictions)


#  4. main(): citire date, discretizare, antrenare, evaluare


def main():
    # Citește fișierul CSV (sen_data.csv)
    df = pd.read_csv('sen_data.csv')
    df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y %H:%M', errors='coerce')
    df.sort_values('Data', inplace=True)
    df.dropna(inplace=True)

    # Extrage Year și Month
    df['Year'] = df['Data'].dt.year
    df['Month'] = df['Data'].dt.month

    # Exclude datele din decembrie 2024 pentru antrenare
    train_df = df[~((df['Year'] == 2024) & (df['Month'] == 12))]
    test_df  = df[(df['Year'] == 2024) & (df['Month'] == 12)]

    feature_cols = [
        'Consum[MW]', 'Carbune[MW]', 'Hidrocarburi[MW]',
        'Ape[MW]', 'Nuclear[MW]', 'Eolian[MW]', 'Foto[MW]', 'Biomasă[MW]'
    ]
    target_col = 'Sold[MW]'

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df[target_col].values

    # Discretizare Sold în 4 intervale
    bins_sold = [-99999, -1000, 0, 1000, 999999]
    labels_sold = ['f_negativ', 'negativ', 'pozitiv', 'f_pozitiv']

    y_train_disc = pd.cut(y_train, bins=bins_sold, labels=labels_sold)
    y_test_disc  = pd.cut(y_test,  bins=bins_sold, labels=labels_sold)

    # Discretizare feature-uri (3 bin-uri fiecare)
    def discretize_column(col_values, n_bins=3, col_name=''):
        """
        Împarte valorile în n_bins intervale egale între min și max.
        Returnează etichete discrete.
        """
        minv = np.min(col_values)
        maxv = np.max(col_values)
        if minv == maxv:
            edges = [minv-1, minv+1]
            lbls = [f"{col_name}_unique"]
            return pd.cut(col_values, bins=edges, labels=lbls, include_lowest=True)
        edges = np.linspace(minv, maxv, n_bins+1)
        lbls = [f"{col_name}_bin{i}" for i in range(n_bins)]
        return pd.cut(col_values, bins=edges, labels=lbls, include_lowest=True)

    df_train_feats = pd.DataFrame()
    df_test_feats  = pd.DataFrame()

    for c in feature_cols:
        train_bin = discretize_column(train_df[c].values, n_bins=3, col_name=c)
        test_bin  = discretize_column(test_df[c].values,  n_bins=3, col_name=c)
        df_train_feats[c] = train_bin
        df_test_feats[c]  = test_bin

    X_train_disc = df_train_feats.values
    X_test_disc  = df_test_feats.values
    feature_names = list(df_train_feats.columns)

    # 4.1 Antrenare ID3
    id3_model = ID3Classifier()
    id3_model.fit(X_train_disc, y_train_disc, feature_names)
    y_pred_id3_disc = id3_model.predict(X_test_disc)

    # Convertirea categoriilor la valori numerice
    center_map = {
        'f_negativ': -1500,
        'negativ':   -500,
        'pozitiv':    500,
        'f_pozitiv':  1500
    }
    y_pred_id3_reg = np.array([center_map[c] for c in y_pred_id3_disc])

    rmse_id3 = math.sqrt(mean_squared_error(y_test, y_pred_id3_reg))
    mae_id3  = mean_absolute_error(y_test, y_pred_id3_reg)

    print("=== Rezultate ID3 ===")
    print("RMSE =", rmse_id3)
    print("MAE  =", mae_id3)

    # 4.2 Antrenare Naive Bayes
    nb_model = NaiveBayesDiscrete()
    nb_model.fit(X_train_disc, y_train_disc, feature_names)
    y_pred_nb_disc = nb_model.predict(X_test_disc)

    y_pred_nb_reg = np.array([center_map[c] for c in y_pred_nb_disc])
    rmse_nb = math.sqrt(mean_squared_error(y_test, y_pred_nb_reg))
    mae_nb  = mean_absolute_error(y_test, y_pred_nb_reg)

    print("=== Rezultate Naive Bayes ===")
    print("RMSE =", rmse_nb)
    print("MAE  =", mae_nb)

    # Concluzii finale
    print("\n===== Concluzii =====")
    print("ID3 -> RMSE:", rmse_id3, "| MAE:", mae_id3)
    print("NB  -> RMSE:", rmse_nb,  "| MAE:", mae_nb)
    print("Au fost discretizate atât feature-urile cât și Sold-ul.")
    print("Datele din decembrie 2024 au fost excluse la antrenare.")
    print("Sfârșit.")

if __name__ == "__main__":
    main()
