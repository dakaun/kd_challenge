import IOHandler as io
import PreprocessingHandler as ph
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB as bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.feature_extraction.text import CountVectorizer as countvec

# bereits probiert: bayes und decision tree für die beiden bag of words
# todo: mit dem metamodell rumspielen und unterschiedliche kombis und 
#       classifier ausprobieren


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

#decision tree für title
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

#decision tree für overview
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