import pandas
import numpy
from matplotlib.pyplot import legend, show, scatter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def reduce(df):
    df.columns = ['Class Label', 'Number of times pregnant', 'Plasma glucose concentration',
                  'Diastolic blood pressure(mm Hg)', 'Triceps skin fold thickness (mm)', 'Insulin (mu U/ml)',
                  'Body mass index (kg/m2)', 'Diabetes pedigree function', 'Age(yrs)']

    X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    sc = StandardScaler()

    X_train_std = sc.fit_transform(X_train)

    covariant_matrix = numpy.cov(X_train_std.T)

    covariant_matrix[0::5]

    eigen_values, eigen_vectors = numpy.linalg.eig(covariant_matrix)

    eigen_pairs = [(numpy.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]

    eigen_pairs.sort(reverse=True)

    w = numpy.hstack((eigen_pairs[0][1] [:, numpy.newaxis], eigen_pairs[1][1][:, numpy.newaxis]))

    X_train_std[0]

    X_train_std[0].dot(w)

    X_train_pca = X_train_std.dot(w)

    colors = ['b', 'r']
    markers = ['x','o']

    for l, c, m in zip(numpy.unique(y_train), colors, markers):
        scatter(X_train_pca[y_train==l, 0], X_train_pca[y_train == l, l],
                c=c, label=l, marker=m)

    legend(loc="lower left")
    show()

if __name__ == "__main__":
    df_diabetes = pandas.read_csv("diabetes.csv", header=None)
    reduce(df_diabetes)