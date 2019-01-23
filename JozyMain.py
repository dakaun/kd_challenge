import IOHandler as io
import PreprocessingHandler as ph
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import DecisionTree as dt
from sklearn import tree
import MLPClassifier as mlp
import SVMClassifier as svm
import NearestCentroidClassifier as nc
import GNBClassifier as gnb
import NearestNeigborClassification as knn

# df = io.read_data('train')
# # # bag of words for overview and title
# #title_bag=ph.bag_of_words(df['title'])
# #overview_bag=ph.bag_of_words(df['overview'])
# #print (title_bag)
# ph.hot_encode_and_write_all_columns(df,'train')
# io.write_df(df['label'], 'label_train')
# label = pd.read_csv('./data/label_train.csv')
# genres = pd.read_csv('./data/genres_train.csv')
# keywords = pd.read_csv('./data/keywords_train.csv')
# production_companies = pd.read_csv('./data/prod_companies_train.csv')
# production_countries = pd.read_csv('./data/prod_countries_train.csv')
# #release_date = pd.read_csv('./data/release_date_train.csv')
# numerical = pd.read_csv('./data/numerical_df_train.csv')
# spoken_languages = pd.read_csv('./data/spoken_languages_train.csv')

# frames = [label,genres, keywords, production_companies, production_countries, numerical, spoken_languages]
# total_df = ph.combine_df(frames)
# io.write_df(total_df, 'total_df')

#X_full=pd.read_csv('./data/total_df.csv')
#label = X_full.loc[:,'label']
#X = X_full.loc[:, X_full.columns !='label']
#bag of words for overview and title
df = io.read_data('train')
data=ph.extract_numerical(df) #numerical data
y=data.loc[:,'label']
X = data.loc[:, data.columns !='label']
title_bag=ph.bag_of_words(df['title'])
overview_bag=ph.bag_of_words(df['overview'])
X_wText = ph.combine_df([X,title_bag,overview_bag])
X_st=ph.standard(X_wText) #standardize data

#split into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X_st, y)


#X_reduced = pd.DataFrame(ph.dim_reduction_components(X, 10))
#frames = [X_reduced, label]
#data_train=ph.combine_df(frames)

# print(X.shape)
# print(X_reduced.shape)

#d=dt.decision_tree(X_full)
#tree.export_graphviz(d, out_file='tree.dot')

#mlp.mlp_classifier(X_full)
#svm.svm_classifier(X_full)
#nc.nc_classifier(X_full)
#gnb.gnb_classifier(X_full)
#knn.knn_classifier(X_full)








#genres_reduction = ph.dim_reduction_var_exp(trans_genre_df, 0.8)

# Create DF from ndarray
#height, width = genres_reduction.shape
#x = ['Genre_PC_%s' % i for i in range(1, width + 1)]
#genres_reduction_df = pd.DataFrame(data=genres_reduction, columns=x)

#train_dataset = ph.combine_df([ph.extract_numerical(df_reduced), genres_reduction_df])
#print(train_dataset)

#dt.decision_tree(train_dataset)
#mlp.mlp_classifier(train_dataset)
#svm.svm_classifier(train_dataset)
#nc.nc_classifier(train_dataset)
#gnb.gnb_classifier(train_dataset)
#knn.knn_classifier(train_dataset)



#prod_countries_reductions = io.read_data('prod_countries_train')
#ph.dim_reduction_var_exp(prod_countries_reductions, 0.8)