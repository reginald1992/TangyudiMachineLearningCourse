import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, classification_report
import itertools
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("./data/creditcard.csv")
print(data.head())
count_classes = pd.value_counts(data['Class'], sort=True).sort_index()
count_classes.plot(kind='bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()
'''
下采样方式解决样本不平衡问题
'''
data['normAmount'] = StandardScaler().fit_transform(np.asarray(data['Amount']).reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)
print(data.head())

x = data.loc[:, data.columns != 'Class']
y = data.loc[:, data.columns == 'Class']

# Number of data points in the minority class
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)

# Picking the indices of the normal classes
normal_indices = np.array(data[data.Class == 0].index)

# Out of the indices we picked, randomly select "x" number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)

# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
# Under sample dataset
under_sample_data = data.iloc[under_sample_indices, :]
x_undersample = under_sample_data.loc[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'Class']
'''
交叉验证方式解决样本不平衡问题
'''
# Whole dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print('Number transaction train dataset:', len(x_train))
print('Number transaction test dataset:', len(x_test))
print('Total number of transactions:', len(x_train) + len(x_test))

# Undersampled dataset
x_train_undersample, x_test_undersample, y_train_undersample, y_test_undersample = train_test_split(x_undersample,
                                                                                                    y_undersample,
                                                                                                    test_size=0.3,
                                                                                                    random_state=0)
print('Number transaction train dataset:', len(x_train_undersample))
print('Number transaction test dataset:', len(x_test_undersample))
print('Total number of transactions:', len(x_train_undersample) + len(x_test_undersample))


# Recall 查全率 TP/（TP+FN）
def printing_Kfold_scores(x_train_data, y_train_data):
    fold = KFold(len(y_train_data), 5, shuffle=False)

    #    Different C parameters
    c_param_range = [0.01, 0.1, 1, 10, 100]

    result_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
    result_table['C_parameter'] = c_param_range

    #    the k-fold will give 2 lists:train_indices=indices[0],test_indices=indices[1]
    j = 0
    for c_param in c_param_range:
        print('-------------------------------')
        print('C parameter:', c_param)
        print('-------------------------------')
        print('')
        recall_accs = []
        for iteration, indices in enumerate(fold, start=1):
            # call the logistic regression model with a certain C parameter
            lr = LogisticRegression(C=c_param, penalty='l1')

            # Use the training data to fit the model. In this case, we use the portion of the fold to train the model
            # with indices[0]. We then predict on the portion assigned as the 'test cross validation' with indices[1]
            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())

            #            Predict values using the test indices in the training data
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)

            # Calculate the recall score and append it to a list for recall scores representing the current c_parameter
            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Iteration', iteration, ': recall score =', recall_acc)

        #        The mean value of those recall scores is the metric we want to save and get hold of
        result_table.ix[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score', np.mean(recall_accs))
        print('')

    best_c = result_table
    best_c.dtypes.eq(object)  # you can see the type of best_c
    new = best_c.columns[best_c.dtypes.eq(object)]  # get the object column of the best_c
    best_c[new] = best_c[new].apply(pd.to_numeric, errors='coerce', axis=0)  # change the type of object
    best_c = result_table.loc[result_table['Mean recall score'].idxmax()]['C_parameter']  # calculate the mean values

    # best_c = result_table.iloc[result_table['Mean recall score'].idxmax()]['C_parameter']

    #    Finally, we can check which C parameter is the best amongst the chosen.
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')

    return best_c


best_c = printing_Kfold_scores(x_train_undersample, y_train_undersample)


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots confusion matrix
    :param cm:
    :param classes:
    :param title:
    :param cmap:
    :return:
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


lr = LogisticRegression(C=best_c, penalty='l1')
lr.fit(x_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample = lr.predict(x_test_undersample.values)

# Compute confusion matrix:下采样后的数据集
cnf_matrix = confusion_matrix(y_test_undersample, y_pred_undersample)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset:", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

# plot non-normalized confusion matrix
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()

y_pred = lr.predict(x_test.values)

# Compute confusion matrix：原始数据集
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()

best_c = printing_Kfold_scores(x_train, y_train)
lr = LogisticRegression(C=best_c, penalty='l1')
lr.fit(x_train, y_train.values.ravel())
y_test_pred = lr.predict(x_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_test_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset:", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion Matrix')
plt.show()

lr = LogisticRegression(C=0.01, penalty='l1')
lr.fit(x_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample_proba = lr.predict_proba(x_test_undersample.values)
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
plt.figure(figsize=(10, 10))
j = 1
for i in thresholds:
    y_test_prediction_high_recall = y_pred_undersample_proba[:, 1] > i
    plt.subplot(3, 3, j)
    j += 1
    #    compute confusion matrix
    cnf_matrix = confusion_matrix(y_test_undersample, y_test_prediction_high_recall)
    np.set_printoptions(precision=2)

    print("Recall metric in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

    # Plot non-normalized confusion matrix

    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Threshold >=%s' % i)

credit_cards = pd.read_csv('./data/creditcard.csv')

columns = credit_cards.columns
# The labels are in the last column ('Class'). Simply remove it to obtain features columns
features_columns = columns.delete(len(columns) - 1)

features = credit_cards[features_columns]
labels = credit_cards['Class']

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2,
                                                                            random_state=0)
oversampler = SMOTE(random_state=0)
os_features, os_labels = oversampler.fit_sample(features_train, labels_train)
print(len(os_labels[os_labels == 1]))
os_features = pd.DataFrame(os_features)
os_labels = pd.DataFrame(os_labels)
best_c = printing_Kfold_scores(os_features, os_labels)

lr = LogisticRegression(C=best_c, penalty='l1')
lr.fit(os_features, os_labels.values.ravel())
y_pred = lr.predict(features_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(labels_test, y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()
