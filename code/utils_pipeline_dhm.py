import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from optbinning import OptimalBinning

def get_woe_mapping(woe_encoder):
    cat_ord = woe_encoder.ordinal_encoder.category_mapping
    ord_woe = woe_encoder.mapping

    cat_ord = {dict_ord['col']:[{"cat": idx, "ord_enc": val} for idx, val in dict_ord['mapping'].items()]
            for dict_ord in cat_ord}

    ord_woe = {variable:[{'ord_enc':ord_enc, 'woe_enc':woe_enc} for ord_enc, woe_enc in woe_dict.items()]
            for variable,woe_dict in ord_woe.items()}

    dfs = []

    for feature in ord_woe.keys():
        df_woe = pd.DataFrame(ord_woe[feature])
        df_cat = pd.DataFrame(cat_ord[feature])
        
        merged = df_cat.merge(df_woe, on='ord_enc', how='left')
        merged.insert(0, 'feature', feature)  # add feature name as first column
        dfs.append(merged)

    df_final = pd.concat(dfs, ignore_index=True)
    df_final = df_final[df_final['cat'].notna()]
    df_final['woe_enc'] = -df_final['woe_enc']
    return df_final

def get_bin_woe_mapping(bin_woe_encoder):
    dfs = []
    for feature, mapping in bin_woe_encoder.mapping.items():

        feature_map = pd.DataFrame(mapping)
        feature_map.insert(0, 'feature', feature)
        dfs.append(feature_map)

    bin_woe_mapping_df = pd.concat(dfs, ignore_index=True)
    return bin_woe_mapping_df


# Custom transformer for optimal binning
class MultiOptimalBinningWOE(BaseEstimator, TransformerMixin):
    def __init__(self, binning_config):
        self.binning_config = binning_config
        self.binners = {}

    def fit(self, X, y):
        self.feature_names_in_ = list(X.columns)  # IMPORTANT
        X = X.copy()

        for col, max_bins in self.binning_config.items():
            optb = OptimalBinning(
                name=col,
                dtype="numerical",
                max_n_bins=max_bins
            )
            optb.fit(X[col], y)
            self.binners[col] = optb

        return self

    def transform(self, X):
        X = X.copy()

        for col, optb in self.binners.items():
            X[col] = optb.transform(X[col])

        return X
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_
    
    @property
    def mapping(self):
        mappings = {}

        for col, optb in self.binners.items():
            table = optb.binning_table.build()

            mappings[col] = (table[["Bin", "WoE","IV"]]
                            .loc[table["Bin"].astype(str).str.strip() != ""]  # drop blank rows
                            .to_dict("records")
                            )

        return mappings    

# Custom transformer for type casting
class TypeCaster(BaseEstimator, TransformerMixin):
    def __init__(self, columns, dtype):
        self.columns = columns
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columns] = X[self.columns].astype(self.dtype)
        return X
    