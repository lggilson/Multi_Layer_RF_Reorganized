import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, accuracy_score


class DecisionNode:
    def __init__(self, col, split, lchild, rchild, left_cat_values = None):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild
        self.left_cat_values = left_cat_values

    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if self.split == -1:
            #categorical split
            return(
                self.lchild.predict(x_test)
                if x_test[self.col] in self.left_cat_values
                else self.rchild.predict(x_test)
            )
        else:
            #numerical split
            return (
                self.lchild.predict(x_test)
                if x_test[self.col] < self.split
                else self.rchild.predict(x_test)
            )

    def leaf(self, x_test):
        """
        Given a single test record, x_test, return the leaf node reached
        by running it down the tree starting at this node.
        This is just like prediction, except we return the decision tree
        leaf rather than the prediction from that leaf.
        """
        return (
            self.lchild.leaf(x_test)
            if x_test[self.col] < self.split
            else self.rchild.leaf(x_test)
        )


class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.y = y
        self.n = len(y)
        self.prediction = prediction

    def predict(self, x_test):
        # return prediction
        return self.prediction

    def leaf(self, x_test):
        """
        Return itself.
        """
        return self


def gini(x):
    """
    Return the gini impurity score for values in y
    Assume y = {0,1}
    Gini = 1 - sum_i p_i^2 where p_i is the proportion of class i in y
    """
    _, class_counts = np.unique(x, return_counts=True)
    n = np.sum(class_counts)

    return 1 - np.sum((class_counts / n) ** 2)

#given categorical labels (numerical but no ordinality), output resorted x and y arrays
def cat_sort(X,y):
    label_index = 0
    unique_labels, inv = np.unique(X, return_inverse=True)
    label_means = np.bincount(inv, weights=y) / np.bincount(inv)
    
    sorted_labels = unique_labels[np.argsort(label_means)]
    expanded_means = label_means[inv]
    mean_sort = np.argsort(expanded_means)

    sorted_x = X[mean_sort]
    sorted_y = y[mean_sort]

    return sorted_x, sorted_y


'''
find best split will do two things, depending on force_label_split. If true, the function will force the tree to split by label at label_depth. 
If false, it will function as normal, but will begin to test splitting by label after label_depth 
For my purposes, the label column will be at label_index (currently the last column); Labels are integer values (Study 0,1,2...n)
Splitting by label attempts to de order the labels before testing splits so as to not imply ordinality
'''

def find_best_split(X, y, loss, min_samples_leaf, max_features, current_depth, label_depth, force_label_split):
    #Removes the label feature from allowed feature space
    label_index = X.shape[1] - 1
    feature_indices = np.arange(X.shape[1])
    feature_indices = feature_indices[feature_indices != label_index]

    # case when this node cannot be further split
    if len(X) <= min_samples_leaf or y.min() == y.max():
        return (-1, -1, loss(y), -1, None)

    '''
    # FORCE label split if we reached label_depth
    if label_depth is not None and current_depth == label_depth and force_label_split:
        split_values = np.unique(X[:, label_index])

        best_loss = (label_index, split_values[0], loss(y), None, None)
        for split in split_values:
            yl = y[X[:, label_index] < split]
            yr = y[X[:, label_index] >= split]

            if len(yl) <= min_samples_leaf or len(yr) <= min_samples_leaf:
                continue

            total_loss = (len(yl) * loss(yl) + len(yr) * loss(yr)) / len(y)

            if total_loss < best_loss[2]:
                best_loss = (label_index, split, total_loss, None, None)

        return best_loss[:-1]

    '''

    # Normal Splitting algorithm
    # record current feature, split and loss, as well as a left and right mask

    # (col, split, loss, cat_boundary, cat_values)
    best_loss = (-1, -1, loss(y), -1, None)

    #Randomly chooses from all features minus the label index. Function breaks if max_features is too low or above 1.
    k = 11
    for i in np.random.choice(
        feature_indices, size=int(max_features * (label_index)), replace=False
    ):
        # randomly pick k values
        idx_splits = np.random.choice(np.arange(len(X)), size=k)
        candidates = X[idx_splits, i].copy()


        for split in np.unique(candidates):
            yl = y[X[:, i] < split]
            yr = y[X[:, i] >= split]

            if len(yl) == 0 or len(yr) == 0:
                continue

            if len(yl) < min_samples_leaf or len(yr) < min_samples_leaf:
                continue

            total_loss = (len(yl) * loss(yl) + len(yr) * loss(yr)) / (len(yl) + len(yr))
            if total_loss == 0:
                return i, split, total_loss, -1, None

            if total_loss < best_loss[2]:
                best_loss = (i, split, total_loss, None, None)


    """
    If force_label_split is False, each split willl test splitting on label feature beginning at label depth
    Effectively this is the declassification after label_depth splits
    This step also works for categorical strings (labels) using CART categorical splits
    """
    if label_depth is not None and current_depth >= label_depth and not force_label_split:
        sorted_x, sorted_y = cat_sort(X[:,label_index],y)
        boundaries = np.where(np.diff(sorted_x) != 0)[0] + 1

        for b in boundaries:

            if b == 0 or b == len(sorted_x):
                continue

            yleft = sorted_y[:b]
            yright = sorted_y[b:]

            xleft = sorted_x[:b]

            left_categories = np.unique(sorted_x[:b])
            right_categories = np.unique(sorted_x[b:])

            if len(left_categories) <= min_samples_leaf or len(right_categories) <= min_samples_leaf:
                continue

            total_loss = (len(yleft) * loss(yleft) + len(yright) * loss(yright)) / (len(yleft) + len(yright))
            if total_loss < best_loss[2]:
                #recording the split as -1 will be used in a function later
                best_loss = (label_index, -1, total_loss, b, None)


        # If no valid split found, don't modify best_loss
        if best_loss[3] == -1:
            return (-1, None, None)  # or return None
        
        xleft = sorted_x[:best_loss[3]]
        #mask_left = np.isin(X, xleft)
        #mask_right = ~mask_left

        best_loss = (
            best_loss[0],
            best_loss[1],
            best_loss[2],
            best_loss[3],
            xleft,
        )
    return best_loss


class DecisionTree621:
    def __init__(self, max_features, min_samples_leaf=1, loss=None, max_depth=None, label_depth=0, force_label_split=False):
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss  # loss function; either np.var for regression or gini for classification

        self.max_depth = max_depth
        self.label_depth = label_depth #depth to declassify - allow find best split to begin accessing last column
        self.force_label_split = force_label_split #this parameter decides whether the model will force a split at label_depth, or just add it to tested splits

    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for  either a classifier or regression.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressions predict the average y
        for observations in that leaf.

        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y, current_depth = 0) #initializes the depth to 0c

    def fit_(self, X, y, current_depth):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classification or regression.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621.create_leaf() depending
        on the type of self.

        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """

        #terminate at max depth
        if self.max_depth is not None and current_depth >= self.max_depth:
            return self.create_leaf(y)

        
        best_split = find_best_split(
            X, y, self.loss, self.min_samples_leaf, self.max_features, current_depth, self.label_depth, self.force_label_split
        )

        col, split,  = best_split[0], best_split[1]
        
        # terminating conditions
        if col == -1:
            return self.create_leaf(y) 
        if best_split == None:
            return self.create_leaf(y)
        
        # mask condition. We record split as -1 for labels because there is a mask rather than a single split
        if split == -1:
            mask_left = np.isin(X[:, col], best_split[4])
            mask_right = ~mask_left

            XL, yl = X[mask_left], y[mask_left]
            XR, yr = X[mask_right], y[mask_right]
        else:
            XL, yl = X[X[:, col] < split], y[X[:, col] < split]
            XR, yr = X[X[:, col] >= split], y[X[:, col] >= split]

        if len(yl) == 0 or len(yr) == 0:
            return self.create_leaf(y)
        
        #fit_ now takes a current depth parameter that we increase with each recursion
        lchild = self.fit_(XL, yl, current_depth + 1)
        rchild = self.fit_(XR, yr, current_depth + 1)

        #best_split[4] is the IDs of the left categories
        return DecisionNode(col, split, lchild, rchild, best_split[4])

    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        return np.array([self.root.predict(x_test) for x_test in X_test])


class RegressionTree621(DecisionTree621):
    def __init__(self, max_features, min_samples_leaf=1, max_depth=None, label_depth=0, force_label_split=False):
        super().__init__(max_features, min_samples_leaf, loss=np.var, max_depth=max_depth, label_depth=label_depth, force_label_split=force_label_split)

    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        return r2_score(y_test, self.predict(X_test))

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))


class ClassifierTree621(DecisionTree621):
    def __init__(self, max_features, min_samples_leaf=1, max_depth=None, label_depth=0, force_label_split=False):
        super().__init__(max_features, min_samples_leaf, loss=gini, max_depth=max_depth, label_depth=label_depth, force_label_split=force_label_split)

    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        return accuracy_score(y_test, self.predict(X_test))

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor. Feel free to use scipy.stats to use the mode function.
        """
        vals, counts = np.unique(y, return_counts=True)
        prediction = vals[np.argmax(counts)]
        return LeafNode(y, prediction)