import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    df = pd.read_csv('./processed_dataset.csv')
    print(df.shape)
    # split X, y
    X = df.drop('Severity4', axis=1)
    y = df['Severity4']

    # split train, test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # Randomly under-sample
    rus = RandomUnderSampler(ratio=0.1, random_state=42)
    X_train_res, y_train_res = rus.fit_sample(X_train, y_train)
    print("Original ratio of other class and Severity5 {}".format(Counter(y_train)))
    print("Under-sampled ratio of other class and Severity5 {}".format(Counter(y_train_res)))

    # LR
    clf_base = LogisticRegression()
    grid = {'C': 10.0 ** pd.np.arange(-2, 3),
            'penalty': ['l1', 'l2'],
            'class_weight': ['balanced']}
    clf_lr = GridSearchCV(clf_base, grid, cv=5, n_jobs=-1, scoring='f1_macro')
    clf_lr.fit(X_train_res, y_train_res)
    coef = clf_lr.best_estimator_.coef_
    intercept = clf_lr.best_estimator_.intercept_
    print(classification_report(y_test, clf_lr.predict(X_test)))
    y_pred = clf_lr.predict(X_test)
    conf_matrix_lr = confusion_matrix(y_true=y_test, y_pred=y_pred)
    conf_matrix = pd.DataFrame(data=conf_matrix_lr,
                               columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
    plt.figure(figsize=(8, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu").set_title("Confusion Matrix \n Logistic Regression", fontsize=14)
    plt.show()

    # RF
    clf_base = RandomForestClassifier()
    grid = {'n_estimators': [10, 50, 100],
            'max_features': ['auto', 'sqrt']}
    clf_rf = GridSearchCV(clf_base, grid, cv=5, n_jobs=-1, scoring='f1_macro')
    clf_rf.fit(X_train_res, y_train_res)
    y_pred = clf_rf.predict(X_test)
    print(classification_report(y_test, y_pred))
    conf_matrix_rf = confusion_matrix(y_true=y_test, y_pred=y_pred)
    conf_matrix = pd.DataFrame(data=conf_matrix_rf,
                               columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
    plt.figure(figsize=(8, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu").set_title("Confusion Matrix \n Random Forest", fontsize=14)
    plt.show()
