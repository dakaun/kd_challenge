# The EvaluationHandler contains methods for calculating and measuring the ML techniques accuracy
import matplotlib.pyplot as plt

def pca_var_exp_visualizer(pca):
    var_exp = pca.explained_variance_ratio_.cumsum()
    x = ['PC %s' % i for i in range(1, pca.n_components_+1)]

    plt.bar(x=x, height=pca.explained_variance_ratio_)
    plt.show()
    plt.bar(x=x, height=var_exp)
    plt.show()

    print(var_exp)