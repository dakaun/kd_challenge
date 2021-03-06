import IOHandler as io
import PreprocessingHandler as ph
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import DecisionTree as dt
from sklearn import tree
import MLPClassifier as mlp
import SVMClassifier as svm
import NearestCentroidClassifier as nc
import GNBClassifier as gnb
import NearestNeigborClassification as knn
from sklearn.naive_bayes import MultinomialNB as bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.feature_extraction.text import CountVectorizer as countvec

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
X_st=ph.standard(X) #standardize data
title=df.loc[:,'title']
#ph.bag_of_words(df['title'])
overview=df.loc[:,'overview']
# ph.bag_of_words(df['overview'])
X_wText = ph.combine_df([X,title,overview,y])

#metamodell
#split into subtrain, metatrain and test
trainSub, tempData = train_test_split(X_wText, test_size=0.4)
trainMeta, test = train_test_split(tempData, test_size=0.4)

#bag of title
trainSub_title = trainSub.loc[:,'title']
trainMeta_title = trainMeta.loc[:,'title']
test_title = test.loc[:,'title']
#bag of overview
trainSub_over = trainSub.loc[:,'overview']
trainMeta_over = trainMeta.loc[:,'overview']
test_over = test.loc[:,'overview']

#numerical data
trainSub_num = trainSub.loc[:,['budget', 'popularity', 'revenue', 'runtime']]
trainMeta_num = trainMeta.loc[:,['budget', 'popularity', 'revenue', 'runtime']]
test_num = test.loc[:,['budget', 'popularity', 'revenue', 'runtime']]
#y
y_trainSub = trainSub['label']
y_trainMeta = trainMeta['label']
y_test = test['label']

#bayes für title
#countvec für title
countvectorizer_title = countvec()
x_trainSub_title = countvectorizer_title.fit_transform(trainSub_title)
x_trainMeta_title = countvectorizer_title.transform(trainMeta_title)
x_test_title = countvectorizer_title.transform(test_title)

tree_title = DecisionTreeClassifier()
tree_title.fit(x_trainSub_title, y_trainSub)
title_score = tree_title.score(x_test_title, y_test)
titleScore_text = "Title Score is {:0.2%}".format(title_score)
print(titleScore_text)

#stack input für meta modell
stacked_input1 = pd.Series(tree_title.predict(x_trainMeta_title))
stacked_input1_test = pd.Series(tree_title.predict(x_test_title))

#bayes für overview
#countvec für overview
countvectorizer_over = countvec()
x_trainSub_over = countvectorizer_over.fit_transform(trainSub_over)
x_trainMeta_over = countvectorizer_over.transform(trainMeta_over)
x_test_over = countvectorizer_over.transform(test_over)

tree_over = DecisionTreeClassifier()
tree_over.fit(x_trainSub_over, y_trainSub)
over_score = tree_over.score(x_test_over, y_test)
overScore_text = "Overview Score is {:0.2%}".format(over_score)
print(overScore_text)

#stack input für meta modell
stacked_input2 = pd.Series(tree_over.predict(x_trainMeta_over))
stacked_input2_test = pd.Series(tree_over.predict(x_test_over))

#knn für numeric daten
#bayes für bag of words
Knn = KNeighborsClassifier(n_neighbors=70)
Knn.fit(trainSub_num, y_trainSub)
num_score = Knn.score(test_num, y_test)
numScore_text = "Num Score is {:0.2%}".format(num_score)
print(numScore_text)

#stack input für meta modell
stacked_input3 = pd.Series(Knn.predict(trainMeta_num))
stacked_input3_test = pd.Series(Knn.predict(test_num))

#meta modell
# initialize RF classifier
forest = RandomForestClassifier()
# build a pandas df for training and one for testing
# meta training data
meta_data_train = {'input1': stacked_input1, 'input2': stacked_input2,'input3': stacked_input3}
meta_data_train = pd.DataFrame(meta_data_train)
print(meta_data_train.iloc[0:10])

# meta test data
meta_data_test = {'input1': stacked_input1_test, 'input2': stacked_input2_test,'input3': stacked_input3_test}
meta_data_test = pd.DataFrame(meta_data_test)
print(meta_data_test.iloc[0:10])

# lets fit the model
forest.fit(meta_data_train, y_trainMeta)

metaScore = forest.score(meta_data_test, y_test)
metaScore_text = "Meta Score is {:0.2%}".format(metaScore)

print(titleScore_text)
print(overScore_text)
print(numScore_text)
print(metaScore_text)

#split into training and testing set
#X_train, X_test, y_train, y_test = train_test_split(X_wText, y)

#pca
# X_train_p=ph.dim_reduction_components(X_train,10)
# X_test_p=ph.dim_reduction_components(X_test,10)

# mlp.mlp_classifier(X_train_p, X_test_p, y_train,y_test)
# svm.svm_classifier(X_train_p, X_test_p, y_train,y_test)
# nc.nc_classifier(X_train_p, X_test_p, y_train,y_test)
# gnb.gnb_classifier(X_train_p, X_test_p, y_train,y_test)
# knn.knn_classifier(X_train_p, X_test_p, y_train,y_test)


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