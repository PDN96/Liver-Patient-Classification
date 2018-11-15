from copy import deepcopy
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.feature_selection import SelectFromModel
import numpy as np
import csv
import pandas as pd
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def read_file(filename):
    df = pd.read_csv(filename, header = None)
    df[1], label1 = pd.factorize(df[1])
    return df

df = read_file('ILPD.csv')
print(df.head(5))

def stratified_sampling(df, class1_p, class2_p):
    df_yes = df[df[10] == 1]
    df_no = df[df[10] == 2]
    df_yes[11] = np.random.uniform(0, 1, len(df_yes)) <= float(class1_p / 10)
    df_no[11] = np.random.uniform(0, 1, len(df_no)) <= float(class2_p/ 10)
    train = pd.DataFrame()
    test = pd.DataFrame()
    train1 = pd.DataFrame()
    train2 = pd.DataFrame()
    test1 = pd.DataFrame()
    test2 = pd.DataFrame()
    train1, test1 = df_yes[df_yes[11] == True], df_yes[df_yes[11] == False]
    train2, test2 = df_no[df_no[11] == True], df_no[df_no[11] == False]
    train = train1.append(train2)
    test = test1.append(test2)
    return train, test

def calc_metrics(pred, test):
    tp = tn = fp = fn = 0
    total = 0
    for x in test[10]:
        if x == pred[total]:
            if x == 1:
                tp = tp + 1
            else:
                tn = tn + 1

        else:
            if x == 1 and pred[total] == 2:
                fn = fn + 1
            if x == 2 and pred[total] == 1:
                fp = fp + 1
        total = total + 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)
    precision = tp / (tp + fp)
    
    return accuracy, sensitivity, specificity, precision

def get_pred(train, test, kernel):
    features = df.columns[0:10]
    y = train[10]
    C = 1.0
    clf = svm.SVC(kernel = kernel, gamma = 0.7, C=C)
    clf.fit(train[features], y)
    pred = clf.predict(test[features])
    
    return pred

def get_all_metrics(kernel, runs):
    all_acc = list()
    all_sens = list()
    all_spec = list()
    all_prec = list()
    for i in range(4, 10):
        acc_i = list()
        sens_i = list()
        spec_i = list()
        prec_i = list()
        for j in range(4, 10):
            acc = list()
            sens = list()
            spec = list()
            prec = list()
            for k in range(runs):
                train, test = stratified_sampling(df, i, j)
                pred = get_pred(train, test, kernel)
                a, s, sp, p = calc_metrics(pred, test)
                #print(acc, sens, spec, prec)
                acc.append(a)
                sens.append(s)
                spec.append(sp)
                prec.append(p)
            acc_i.append(np.mean(acc))
            sens_i.append(np.mean(sens))
            spec_i.append(np.mean(spec))
            prec_i.append(np.mean(prec))
        all_acc.append(acc_i)
        all_sens.append(sens_i)
        all_spec.append(spec_i)
        all_prec.append(prec_i)
    return all_acc, all_sens, all_spec, all_prec  

i = [40,50,60,70,80,90]
j = [40,50,60,70,80,90]

a, s, sp, pr = get_all_metrics('rbf', 10)

max_rbf = 0
for k in range(0,6):
    for l in range(0,6):
        a[k][l] = a[k][l]*100
        if a[k][l]>max_rbf:
            max_rbf = a[k][l]

for k in range(0,6):
        plt.plot(i, a[k], 'o-', label="Class 1 samples in training data = " +str(i[k])+ "%")
plt.ylabel("Accuracy")
plt.xlabel("Percentage of class 2 samples in training data")
plt.legend()
plt.savefig("acc.png")

print("Maximum accuracy achieved with rbf kernel: ",max_rbf,"%")