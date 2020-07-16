from scipy import stats
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import LocallyLinearEmbedding, TSNE
import pandas as pd
import numpy as np

class DataProcessing:
    def load_csv(self, file_path, delimiter):
        self.__df = pd.read_csv(filepath_or_buffer=file_path, delimiter=delimiter)

    def load_excel(self, file_path, header, sheet_name):
        self.__df = pd.read_excel(io=file_path, header=header, sheet_name=sheet_name)
    
    def drop_missing_rate(self, threshold):
        self.__df = self.__df.loc[:, self.__df.isnull().sum() < threshold * self.__df.shape[0]]

    def drop_type(self, type_to_drop):
        self.__df = self.__df.select_dtypes(exclude=[type_to_drop])

    def drop_nonunique(self, threshold):
        non_unique = self.__df.apply(pd.Series.nunique)
        cols_to_drop = non_unique[non_unique <= (self.__df.shape[0] * threshold * 10)/100].index
        self.__df = self.__df.drop(cols_to_drop, axis=1)
    
    def drop_duplicated(self):
        self.__df = self.__df.T.drop_duplicates().T

    def drop_zeros(self, threshold):
        self.__df = self.__df.loc[:, self.__df.isin(['', 'NULL', 'NaN', 0]).mean() < threshold]

    def fill_missing(self, fill_with):
        func = {"median": self.__df.median(), "mean": self.__df.mean()}
        self.__df = self.__df.fillna(func[fill_with])

    def get_data_types(self):
        return self.__df.dtypes

    def to_float(self):
        self.__df = self.__df.astype(np.float64)

    def to_object(self):
        self.__df = self.__df.astype(np.object)

    def label_encoding(self):
        X_cat = self.__df.select_dtypes(include=['object'])
        X_categorical_cols = X_cat.columns
        X_cat = X_cat.apply(LabelEncoder().fit_transform) 
        X_cat = X_cat[X_categorical_cols]
        self.__df = pd.concat([self.__df, X_cat], axis=1)

    def column_type_change(self, column, desired_type):
        self.__df[column] = self.__df[column].astype(np.desired_type)

    def calculate_zscore(self):
        self.__z_score = stats.zscore(self.__df)

    def removeoutlier_zscore(self, z_min, z_max):
        self.__z_score = stats.zscore(self.__df)
        self.__df = self.__df[(stats.zscore(self.__df) < z_min and stats.zscore(self.__df) > z_max).all(axis=1)]
    
    def removeoutlier_stddev(self, threshold):
        self.__df = self.__df[np.abs(self.__df-self.__df.mean()) <= (threshold*self.__df.std())]

    def get_zscore(self, z_min, z_max):
        return np.where(self.__z_score  > z_max and self.__z_score < z_min)

    def fill_outlier(self, threshold, fill_with):
        func = {"median": self.__df.median(), "mean": self.__df.mean()}
        self.__df = self.__df.mask(self.__df.sub(func).div(self.__df.std()).abs().gt(threshold))
        self.__df = self.__df.fillna(func)

    def one_hot_encode(self):
        self.__df = pd.get_dummies(self.__df)

    def delete_rows(self, min, max, selected_column):
        c_min = (self.__df[selected_column] <= min) if min != None else True
        c_max = (self.__df[selected_column] >= max) if max != None else True
        self.__df = self.__df[(c_max & c_min)]

    def reindex(self):
        indices_to_keep = ~self.__df.isin([np.nan, np.inf, -np.inf]).any(1)
        self.__df = self.__df[indices_to_keep]
        self.__df.info()

    def scale(self, scaling_method, transform_method):
        obj = {
            "normalizer_l1": Normalizer(norm='l1'),
            "normalizer_l2": Normalizer(norm='l2'),
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "maxabsscale": MaxAbsScaler(),
            "quantile": QuantileTransformer(),
            "powertransformer": PowerTransformer()
        }

        self.__df = pd.DataFrame(obj.fit(self.__df) if transform_method == "fit" else obj.fit_transform(self.__df), columns=self.__df.columns)

    
    def split(self, target, test_size, random_state, shuffle_status, stratify_status):
        self.__y = self.__df[target]
        self.__X = self.__df.drop([target], axis = 1)

        if stratify_status == None:
            self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X, self.__y, test_size=test_size,
                                                                random_state=random_state, shuffle=shuffle_status)
        else:
            self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X, self.__y, test_size=test_size, stratify=stratify_status,
                                                            random_state=random_state, shuffle=shuffle_status)
    
    def get_Xtrain(self):
        return self.__X_train
    
    def get_Xtest(self):
        return self.__X_test
    
    def get_ytrain(self):
        return self.__y_train
    
    def get_ytest(self):
        return self.__y_test


    def feature_extraction(self, extraction_method, n_components):
        obj = {
            "pca": PCA(n_components=n_components),
            "ica": FastICA(n_components=n_components),
            "lda": LinearDiscriminantAnalysis(n_components=n_components),
            "lle": LocallyLinearEmbedding(n_components=n_components),
            "tsne": TSNE(n_components=n_components, verbose=1, perplexity=40, n_iter=300)
        }

        self.__df = obj.fit(self.__X, self.__y).transform(self.__X)
    
    def polynomial_degree(self, degree,interaction):
        poly = PolynomialFeatures(degree=degree,interaction_only=interaction).fit(self.__df)
        self.__df = poly.transform(self.__df)
    
    def combine_column(self, columns):
        extracted_feature = columns[0]
        calc_column = self.__df[columns[0]]
        for i in range(1, len(columns)):
            extracted_feature += ("x" + str(columns[i]))
            calc_column *= self.__df[columns[i]]

        self.__df[extracted_feature] = calc_column

    def get_correlation(self,target):
        return self.__df.corrwith(self.__df[target])

    def data_summary(self,target):
        summary = {
            "std": self.__df.std(),
            "mean": self.__df.mean(),
            "media": self.__df.median(),
            "min": self.__df.min(),
            "max": self.__df.max(),
            "column_count": len(self.__df.columns),
            "row_count": len(self.__df),
            "missing_values": self.__df.isnull().sum(),
            "datatypes": self.__df.dtypes,
            "quantile": self.__df.quantile([.1, .25, .5, .75, .90], axis = 0),
            "correlation": self.__df.corrwith(self.__df[target])
        }

        return summary

    
    def decision(self, target):
        nununique = self.df[target].nunique()
        return "Classification" if (nununique/len(self.df)) <0.4 else "Regression" 
