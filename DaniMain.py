import PreprocessingHandler
import IOHandler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import DecisionTree as dt
import MLPClassifier as mlp
import SVMClassifier as svm
import NearestCentroidClassifier as nc
import GNBClassifier as gnb
import NearestNeigborClassification as knn

df = pd.read_csv('./data/train.csv')
label = df[['label']]
IOHandler.write_df(label, 'label')

###################################################################################
###### WHOLE PROCESS (INCL Write and read all new files) - das wäre für dich jules
# get all columns
PreprocessingHandler.hot_encode_and_write_all_columns(df, 'train')
# since all df are written als files, read them again to concat
genres = pd.read_csv('./data/genres_train.csv')
keywords = pd.read_csv('./data/keywords_train.csv')
production_companies = pd.read_csv('./data/prod_companies_train.csv')
production_countries = pd.read_csv('./data/prod_countries_train.csv')
#release_date = pd.read_csv('./data/release_date_train.csv')
numerical = pd.read_csv('./data/numerical_df_train.csv')
spoken_languages = pd.read_csv('./data/spoken_languages_train.csv')
label = pd.read_csv('./data/label.csv')
# concat all files to total_df
frames = [genres, keywords, production_companies, production_countries, numerical, spoken_languages, label]
total_df = PreprocessingHandler.combine_df(frames)
# write total_df
IOHandler.write_df(total_df, 'total_df_1')

total_df_ohnelabel_1 = total_df.loc[:, total_df.columns !='label']
# use Pca -- 9 components war von uns getestet, das beste ergebnis
X_reduced_1 = PreprocessingHandler.dim_reduction_components(total_df_ohnelabel_1, n_components=9, visualize= False)
#X_reduced = PreprocessingHandler.dim_reduction_var_exp(total_df_ohnelabel, 0.6)
X_reduced_df_1 = pd.DataFrame(X_reduced_1) # transform to dataframe
# concat and write final file
frames = [X_reduced_df_1, label]
total_df_reduced = PreprocessingHandler.combine_df(frames)
IOHandler.write_df(total_df_reduced, 'total_df_reduced_1')
# jetzt kann jules main durchgelaufen werden

###################################################################################
###### WHOLE PROCESS (READING TOTAL_DF)
wrong_total_df = pd.read_csv('./data/total_df.csv')
total_df_ohnelabel = wrong_total_df.loc[:, wrong_total_df.columns !='label']

X_reduced = PreprocessingHandler.dim_reduction_components(total_df_ohnelabel, n_components=9, visualize= False)
#X_reduced = PreprocessingHandler.dim_reduction_var_exp(total_df_ohnelabel, 0.6)
X_reduced_df = pd.DataFrame(X_reduced)
frames = [X_reduced_df, label]
total_df_reduced = PreprocessingHandler.combine_df(frames)
IOHandler.write_df(total_df_reduced, 'total_df_reduced')

###################################################################################
###### Jules Main
dt.decision_tree(total_df_reduced)
mlp.mlp_classifier(total_df_reduced)
svm.svm_classifier(total_df_reduced)
nc.nc_classifier(total_df_reduced)
gnb.gnb_classifier(total_df_reduced)
knn.knn_classifier(total_df_reduced)

###################################################################################
##############################
#bag of words for overview and title
overview = PreprocessingHandler.bag_of_words(df[['overview']])
overview = pd.DataFrame(overview.toarray())

##############################
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X_reduced_df_1, label)
randomforest = RandomForestClassifier()
randomforest.fit(X_train, y_train)
y_pre = randomforest.predict(X_test)
rf_score = randomforest.score(X_test, y_test)
print(f'Random Forest Score is : {rf_score}')

##############################
#print(genres.shape)
# print(keywords.shape)
# print(production_companies.shape)
# print(production_countries.shape)
# print(spoken_languages.shape)
# print(numerical.shape)
# print(total_df.shape)
