import IOHandler as io
import PreprocessingHandler as ph
import pandas as pd

import DecisionTree as dt
import MLPClassifier as mlp
import SVMClassifier as svm
import NearestCentroidClassifier as nc
import GNBClassifier as gnb
import NearestNeigborClassification as knn

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
mlp.mlp_classifier(train_dataset)
svm.svm_classifier(train_dataset)
nc.nc_classifier(train_dataset)
gnb.gnb_classifier(train_dataset)
knn.knn_classifier(train_dataset)


# Todo IMDB rating crawler (https://datascience.stackexchange.com/questions/5534/how-to-scrape-imdb-webpage)


#prod_countries_reductions = io.read_data('prod_countries_train')
#ph.dim_reduction_var_exp(prod_countries_reductions, 0.8)