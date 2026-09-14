import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, accuracy_score
from numba import njit

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

        if self.split == -1:  
            # categorical
            return (
                self.lchild.leaf(x_test)
                if x_test[self.col] in self.left_cat_values
                else self.rchild.leaf(x_test)
            )
        else:  
            # numeric
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

def cat_sort(X,y):
    '''
    given categorical labels (numerical but no ordinality), output resorted x and y arrays, 
    sorted by mean y for each label. This ordinality speeds up the 2^(feature values) possibilities than 
    non-ordered features represent
    '''
    label_index = 0
    unique_labels, inv = np.unique(X, return_inverse=True)
    label_means = np.bincount(inv, weights=y) / np.bincount(inv)
    
    #sorted_labels = unique_labels[np.argsort(label_means)]
    expanded_means = label_means[inv]
    mean_sort = np.argsort(expanded_means)

    sorted_x = X[mean_sort]
    sorted_y = y[mean_sort]

    return sorted_x, sorted_y

'''
Use this for classification splitting
'''
@njit
def quantile_split_gini(sorted_x, sorted_y, min_samples_leaf, quantiles, current_best_loss):
    loss_val = current_best_loss
    best_split = -1
    n = len(sorted_y)

    for split in quantiles:
        if split == 0 or split == n:
            continue

        left_n  = split
        right_n = n - split

        if left_n <= min_samples_leaf or right_n <= min_samples_leaf:
            continue

        # gini for left
        left_y = sorted_y[:split]
        # can't use np.unique in njit easily, so compute gini manually
        left_ones  = np.sum(left_y)
        left_zeros = left_n - left_ones
        p_left  = left_ones / left_n
        left_gini = 1 - (p_left**2 + (1-p_left)**2)

        # gini for right
        right_y    = sorted_y[split:]
        right_ones = np.sum(right_y)
        right_zeros = right_n - right_ones
        p_right = right_ones / right_n
        right_gini = 1 - (p_right**2 + (1-p_right)**2)

        total_loss = (left_n * left_gini + right_n * right_gini) / n

        if total_loss < loss_val:
            loss_val   = total_loss
            best_split = split

    return best_split, loss_val

'''
find best split will begin to test splitting by label after min_samples_label is reached. 
For my purposes, the label column will be at label_index (currently the last column); Labels are integer values (Study 0,1,2...n)
Splitting by label attempts to de order the labels before testing splits so as to not imply ordinality
'''
@njit
def quantile_split(sorted_x, sorted_y, min_samples_leaf, quantiles, current_best_loss):
    loss_val = current_best_loss
    best_split = -1

    n = len(sorted_y)
    cumsum = np.cumsum(sorted_y)
    cumsum_sq = np.cumsum(sorted_y**2)
    total_sum = cumsum[-1]
    total_sq = cumsum_sq[-1]

    for split in quantiles:
        if split == 0 or split == n:
            continue

        left_n = split
        right_n = n - split

        if left_n <= min_samples_leaf or right_n <= min_samples_leaf:
            continue

        left_sum = cumsum[split - 1]
        right_sum = total_sum - left_sum

        left_sq = cumsum_sq[split - 1]
        right_sq = total_sq - left_sq

        left_var = max(0.0, (left_sq / left_n) - (left_sum / left_n)**2)
        right_var = max(0.0, (right_sq / right_n) - (right_sum / right_n)**2)

        total_loss = (left_n * left_var + right_n * right_var) / n

        if total_loss < loss_val or (np.isclose(total_loss, loss_val) and np.random.rand() < 0.5):
            loss_val = total_loss
            best_split = split

    return best_split, loss_val

def find_best_split(X, y, loss, min_samples_leaf, max_features, current_depth, label_depth, min_samples_label, thick_trunk):
    #Removes the label feature from allowed feature space
    label_index = X.shape[1] - 1
    feature_indices = np.arange(X.shape[1])

    #allowed non label feature columns
    feature_indices = feature_indices[feature_indices != label_index]

    # case when this node cannot be further split
    if len(X) <= min_samples_leaf or y.min() == y.max():
        return (-1, -1, loss(y), -1, None)

    # (col, split number, loss, boundary in sorted space, xleft values)
    best_loss = (-1, -1, loss(y), -1, None)

    #Randomly chooses from all features minus the label index. Function breaks if max_features is too low or above 1.
    k = 64 #tests 64 splits for each feature rather than every possible split

    #If the number of data points at this node is less than min_samples_label begin splitting by label
    if ((min_samples_label is not None and len(X) <= min_samples_label) or
    (label_depth is not None and current_depth < label_depth)) and thick_trunk:
        features = 1
    else:
        features = max_features
    
    size = int(features * (X.shape[1]))
    size = min(size, len(feature_indices))
    for i in np.random.choice(
        feature_indices, size = size, replace=False
    ):
        sorted_idx = np.argsort(X[:, i])
        sorted_x = X[sorted_idx, i]
        sorted_y = y[sorted_idx]

        quantiles = np.linspace(0.0,len(X[:,i]), k+2, dtype=int)[1:-1]

        #this function is compiled with numba for time save
        if loss == np.var:
            split, loss_val = quantile_split(sorted_x,sorted_y, min_samples_leaf, quantiles, best_loss[2])
        else:
            split, loss_val = quantile_split_gini(sorted_x,sorted_y, min_samples_leaf, quantiles, best_loss[2])
        

        if loss_val < best_loss[2]:
            best_loss = (i, sorted_x[split], loss_val, -1, sorted_x[:split])  # keep slice in Python
            if loss_val == 0:
                return best_loss

    """
    Each split will test splitting on label feature beginning at label depth
    Effectively this is the declassification after min_samples_label number of samples in the node
    This step also works for categorical strings (labels) using CART categorical splits
    """
    sorted_x, sorted_y = cat_sort(X[:,label_index],y)
    

    # calculate weights (inverse source size)
    source_sizes = {s: np.sum(X[:, -1] == s) for s in np.unique(X[:, -1])}
    weights = np.array([1.0 / source_sizes[src] for src in sorted_x])

    # prefix sums
    w_cumsum = np.cumsum(weights)
    cumsum = np.cumsum(weights * sorted_y)
    cumsum_sq = np.cumsum(weights * sorted_y**2)

    total_sum = cumsum[-1]
    total_sq = cumsum_sq[-1]
    total_w = w_cumsum[-1]

    if (min_samples_label is not None and len(X) <= min_samples_label) or (label_depth != None and current_depth >= label_depth):
        #sorted_x, sorted_y = cat_sort(X[:,label_index],y)
        boundaries = np.where(np.diff(sorted_x) != 0)[0] + 1
        for b in boundaries:

            if b == 0 or b == len(sorted_x):
                continue
            
            left_n = b
            right_n = len(sorted_y) - b

            if left_n <= min_samples_leaf or right_n <= min_samples_leaf:
                continue

            left_w   = w_cumsum[b-1]
            right_w  = total_w - left_w
            
            left_sum = cumsum[b-1]
            right_sum = total_sum - left_sum

            left_sq = cumsum_sq[b-1]
            right_sq = total_sq - left_sq

            if loss == np.var:
                # weighted variance calculation
                left_var  = max(0.0, (left_sq / left_w) - (left_sum / left_w)**2)
                right_var = max(0.0, (right_sq / right_w) - (right_sum / right_w)**2)
                total_loss = (left_w * left_var + right_w * right_var) / total_w
            else:
                # gini for classification
                left_loss  = loss(sorted_y[:b])
                right_loss = loss(sorted_y[b:])
                total_loss = (left_n * left_loss + right_n * right_loss) / len(sorted_y)
            
            margin = 0
            if total_loss < best_loss[2] * (1 + margin):
                #recording the split as -1 will be used in a function later
                #best_loss = (label_index, -1, total_loss, b, left_categories)
                left_categories = np.unique(sorted_x[:b])
                best_loss = (label_index, -1, total_loss, b, left_categories)
        #mask_left = np.isin(X, xleft)
        #mask_right = ~mask_left
    #print(current_depth)
    return best_loss


class DecisionTree621:
    def __init__(self, max_features, min_samples_leaf=1, loss=None, max_depth=None, label_depth=None, min_samples_label=None, thick_trunk = False):
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss  # loss function; either np.var for regression or gini for classification

        self.max_depth = max_depth
        self.label_depth = label_depth
        self.min_samples_label = min_samples_label #depth to declassify - allow find best split to begin accessing last column
        self.first_label_split_depth = None
        self.thick_trunk = thick_trunk

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
            X, y, self.loss, self.min_samples_leaf, self.max_features, current_depth, self.label_depth, self.min_samples_label, self.thick_trunk
        )

        col, split,  = best_split[0], best_split[1]
        
        # terminating conditions
        if col == -1:
            return self.create_leaf(y) 
        if best_split == None:
            return self.create_leaf(y)
        
        #Masks by either split in sorted space, or boundary in category sorted space.
        if split == -1:
            mask_left = np.isin(X[:, col], best_split[4])
            mask_right = ~mask_left
            #print("unique left:", np.unique(X[mask_left, col]))
            #print("stored left:", best_split[4])
            #print("label split!")

            XL, yl = X[mask_left], y[mask_left]
            XR, yr = X[mask_right], y[mask_right]

            if self.first_label_split_depth == None:
                self.first_label_split_depth = current_depth
                #print(self.first_label_split_depth)
        else:
            XL, yl = X[X[:, col] < split], y[X[:, col] < split]
            XR, yr = X[X[:, col] >= split], y[X[:, col] >= split]


        if len(yl) == 0 or len(yr) == 0:
            return self.create_leaf(y)
        
        #fit_ now takes a current depth parameter that we increase with each recursion
        lchild = self.fit_(XL, yl, current_depth + 1)
        rchild = self.fit_(XR, yr, current_depth + 1)

        #best_split[4] is the values in the left child
        return DecisionNode(col, split, lchild, rchild, best_split[4])

    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        return np.array([self.root.predict(x_test) for x_test in X_test])


class RegressionTree621(DecisionTree621):
    def __init__(self, max_features, min_samples_leaf=1, max_depth=None, label_depth=None, min_samples_label=None, thick_trunk = False):
        super().__init__(max_features, min_samples_leaf, loss=np.var, max_depth=max_depth, label_depth=label_depth, min_samples_label=min_samples_label, thick_trunk=thick_trunk)

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
    def __init__(self, max_features, min_samples_leaf=1, max_depth=None, label_depth=None, min_samples_label=None, thick_trunk=False):
        super().__init__(max_features, min_samples_leaf, loss=gini, max_depth=max_depth, label_depth=label_depth, min_samples_label=min_samples_label, thick_trunk=thick_trunk)


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