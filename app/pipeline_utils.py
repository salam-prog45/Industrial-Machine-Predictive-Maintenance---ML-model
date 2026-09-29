from sklearn.base import BaseEstimator, TransformerMixin


class IQRCapper(BaseEstimator, TransformerMixin):
    """Clips numeric columns to their IQR-based [Q1-1.5*IQR, Q3+1.5*IQR] bounds."""

    def fit(self, X, y=None):
        X = X.copy()
        self.numeric_columns_ = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.lower_bounds_, self.upper_bounds_ = {}, {}
        for col in self.numeric_columns_:
            Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds_[col] = Q1 - 1.5 * IQR
            self.upper_bounds_[col] = Q3 + 1.5 * IQR
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.numeric_columns_:
            X[col] = X[col].clip(lower=self.lower_bounds_[col], upper=self.upper_bounds_[col])
        return X
