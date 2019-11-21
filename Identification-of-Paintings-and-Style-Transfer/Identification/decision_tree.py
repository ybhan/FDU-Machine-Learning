from sklearn import tree


def tree_predict(input_data, input_class, data):
    """Use Decision Tree to predict.
    Args:
        input_data: feature of the picture
        input_class : class of the input
        data: data to be predicted
    Return:
        ans: predicted class of the data
    """
    model = tree.DecisionTreeClassifier()
    model = model.fit(input_data, input_class)
    ans = model.predict(data)

    return ans
