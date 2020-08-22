from sklearn.base import BaseEstimator, TransformerMixin

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(name)s-%(levelname)s: %(message)s',
    handlers=[logging.FileHandler('btc-desafio2.log'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        logger.info(
        f"Dropping the following {len(self.columns)} unusable columns:\n"
        f"{self.columns}"
        )

        data = X.copy()

        # Retornamos um novo dataframe sem as colunas indesejadas
        df = data.drop(labels=self.columns, axis='columns')
        
        logger.info(
        f"Remaining {len(df.columns)} columns:\n {sorted(df.columns.tolist())}"
        )
        return df

class FillNulls(BaseEstimator, TransformerMixin):
    def  __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
        
    #define el tipo de la columna
    def _get_typed_cols(self, col_type ='cat'): 
        assert col_type in ('cat', 'num')
        include = 'object' if col_type == 'cat' else [np.number]
        typed_cols = [
            c for c in self.select_dtypes(include=include).columns
        ]
        return typed_cols
  
    def transform(self, X):
        data = X.copy()
        #df = _fill_nulls(data)
        for t in ['num', 'cat']:
            cols = _get_typed_cols(data, col_type=t)
            for c in cols:
                if t == 'num':
                    data[c] = data[c].fillna(data[c].median())
                else:
                    val_count = data[c].value_counts(dropna=True)
                    common_val = val_count.index.tolist()[0]
                    data[c] = data[c].fillna(common_val)
        return data
