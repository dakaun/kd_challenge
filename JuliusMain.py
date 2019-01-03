import IOHandler as io
import PreprocessingHandler as ph
import pandas as pd
import DecisionTree as dt

df = io.read_data('train')
df_reduced = df[df.imdbId != 233699].reset_index(drop=True)
# Index must be reset to allow for flawless concatination

# Hot encoding using PreProcessing
genre_df = df_reduced['genres']
trans_genre_df = ph.transform_dictionary(genre_df)
genres_reduction = ph.dim_reduction_var_exp(trans_genre_df, 0.8)

# Create DF from ndarray
height, width = genres_reduction.shape
x = ['Genre_PC_%s' % i for i in range(1, width + 1)]
genres_reduction_df = pd.DataFrame(data=genres_reduction, columns=x)

train_dataset = ph.combine_df([ph.extract_numerical(df_reduced), genres_reduction_df])
#print(train_dataset)

dt.decision_tree(train_dataset)

#prod_countries_reductions = io.read_data('prod_countries_train')
#ph.dim_reduction_var_exp(prod_countries_reductions, 0.8)