import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import classification_rf

from collections import namedtuple
Dataset = namedtuple('Dataset', ['X_train', 'X_test', 'y_train', 'y_test'])
Params = namedtuple('Params', ['min_samples_leaf', 'max_features', 'max_depth', 'oob_score'])
Data = namedtuple('Data', ['X','y'])

from classification_rf import RandomForestClassifier621

def generate_study(features, n, study_label, similarity, intercept_sd, signal_strength = 1.0):
    X = np.random.normal(loc = 0, scale = 1, size = (n,features)) #Base random inputs

    #base = np.arange(features, dtype=float)
    i = np.arange(features)
    k = 0.2 #Arbitrary
    base = np.exp(-1*k*i)

    tau = 0.1

    noise = np.random.normal(loc=0, scale=tau, size=features) #inter study weights will vary by this noise parameter controlled by tau
    base  /= np.linalg.norm(base)

    b = similarity * base + (1-similarity)*noise #At similarity 0,1 we get noise and unison.
    b *= signal_strength

    weighted_X = np.dot(X,b)
    
    intercept = np.random.normal(0, intercept_sd)

    pX = np.exp(weighted_X+intercept)/(1+np.exp(weighted_X+intercept)) # interval mapping (-inf,inf) to (0,1)

    y = np.random.binomial(n = 1, p = pX) #Random 0 or 1 with probability pX
    
    print(f"norm of b: {np.linalg.norm(b):.4f}")
    print(f"X shape: {X.shape}")
    print(f"Study {study_label}: intercept={intercept:.2f}, base_rate={y.mean():.2f}")
    print(f"X @ b range: {(X @ b).min():.2f} to {(X @ b).max():.2f}")
    
    labels = [study_label]*n
    X = np.column_stack([X,labels])

    return X,y

def generate_dataset(n_datapoints_per_study, n_studies, features, similarity, intercept_sd, signal_strength = 1.0, random_state = 0):
    np.random.seed(random_state)

    #Generate first study
    X,y = generate_study(
        features = features, 
        n = n_datapoints_per_study, 
        study_label = 0, 
        similarity = similarity, 
        intercept_sd = intercept_sd,
        signal_strength = signal_strength
    )

    #Generate studies 2 through n
    for i in range(1,n_studies):
        X_temp,y_temp = generate_study(
            features = features, 
            n = n_datapoints_per_study, 
            study_label = i, 
            similarity = similarity, 
            intercept_sd = intercept_sd
        )
        X = np.concatenate((X,X_temp), axis = 0)
        y = np.concatenate((y,y_temp), axis = 0)

    X_train, X_test, y_train, y_test = split_by_source(X, y, test_size=0.2, random_state = random_state)
    dataset = Dataset(X_train, X_test, y_train, y_test)
    data = Data(X,y)

    return dataset, data

def forest(n_estimators, dataset, parameters, label_depth=None, min_samples_label=None,  metric = "score", seed = 24):
    
    # In forest():
    X_train = dataset.X_train
    X_test  = dataset.X_test
    y_train = dataset.y_train
    y_test = dataset.y_test

    min_samples_leaf = parameters.min_samples_leaf
    max_features = parameters.max_features
    max_depth = parameters.max_depth
    oob_score = parameters.oob_score

    np.random.seed(seed)
    
    rf1 = RandomForestClassifier621(
        n_estimators=n_estimators, 
        label_depth=label_depth,
        min_samples_label=min_samples_label,
        min_samples_leaf=min_samples_leaf, 
        max_features=max_features, 
        max_depth=max_depth, 
        oob_score=oob_score,
        thick_trunk=False
    )
    #fit the model to the training data
    rf1.fit(X_train, y_train)

    #calculate the accuracy score or auc score for the forest
    if metric == "score":
        accuracy = rf1.score(X_test, y_test)
    if metric == "auc":
        accuracy = rf1.auc_score(X_test, y_test)

    #calculate accuracy for each dataset in Xtest and place into array sourced score
    unique_vals = np.unique(X_test[:, -1])  # get unique values in label column
    source_score = []
    for val in unique_vals:
        sourced_X = X_test[X_test[:, -1] == val]
        sourced_y = y_test[X_test[:, -1] == val]
        if metric == "score":
            source_score.append(rf1.score(sourced_X, sourced_y))
        if metric == "auc":
            source_score.append(rf1.auc_score(sourced_X,sourced_y))

    source_size = [] #size of each source
    for val in unique_vals:
        source_size.append(len(X_train[X_train[:, -1] == val]))

    return accuracy, rf1, source_score, source_size

def forest_wrapper(n_estimators, depth, dataset, parameters, metric = "score", seed = 24, plot = True):
    #mask = X_test[:,-1] == 3
    #print(f"Seed {seed}: Source 3 test indices = {np.where(mask)[0][:5]}")

    accuracy_score = []
    rf = []
    source_score = []

    for i in range(depth):
        #print("forest", i)
        temp_accuracy_score, temp_rf, temp_source_score, source_size = forest( 
            n_estimators, 
            dataset = dataset, 
            parameters=parameters,
            label_depth=i,
            min_samples_label=None, 
            seed = seed, 
            metric = metric)
        
        accuracy_score.append(temp_accuracy_score)
        rf.append(temp_rf)
        source_score.append(temp_source_score)
        
        
        '''
        mask = np.where(dataset.X_test[:,-1] == 1.8048780487804879)
        preds = temp_rf.predict(dataset.X_test[mask])
         
        print(f"Depth {i} Seed {seed}: preds={preds}, true={dataset.y_test[mask]}")
        '''
    
    source_score = np.array(source_score)

    if plot == True:
        depth_range = np.arange(depth)

        for i in range(len(source_score[0])):
            plt.plot(depth_range, source_score[:,i], label = f"Source {i} (n =  {source_size[i]})")

        plt.plot(depth_range, accuracy_score, color = 'black', linestyle = "dashed", label = "Overall Accuracy")
        plt.xlabel('Label Unmask Minimum Samples')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title("Sourced accuracy by D0 depth")
    

    return accuracy_score, rf, source_score, source_size

def msl_forest_wrapper(n_estimators, n_forests, dataset, parameters, metric = "score", seed = 24, plot = True):
    '''
    Creates forests at evenly spaced min_samples_label so that we get entire curve from len(X) to never
    '''
    accuracy_score = []
    rf = []
    source_score = []

    msl_set = np.linspace(len(dataset.X_train), parameters.min_samples_leaf, num=n_forests, dtype=int)

    for min_samples_label in msl_set:
        temp_accuracy_score, temp_rf, temp_source_score, source_size = forest(
            n_estimators, 
            dataset = dataset, 
            parameters = parameters,
            label_depth = None,
            min_samples_label = min_samples_label,
            seed = seed,
            metric = metric
            )
        
        accuracy_score.append(temp_accuracy_score)
        rf.append(temp_rf)
        source_score.append(temp_source_score)
    
    source_score = np.array(source_score)

    if plot == True:
        for i in range(len(source_score[0])):
            plt.plot(msl_set, source_score[:,i], label = f"Source {i} (n =  {source_size[i]})")

        plt.plot(msl_set, accuracy_score, color = 'black', linestyle = "dashed", label = "Overall Accuracy")
        plt.xlabel('Label Unmask Minimum Samples')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title("Sourced accuracy by D0 depth")
        plt.gca().invert_xaxis()
    

    return accuracy_score, rf, source_score, source_size

def errorbars_stochastic_forest(n_seeds, n_estimators, depth, dataset, parameters, metric="score", seed = 24, plot = True):
    forests = [] # forest[0] is forests with same dataset
    accuracy_score = []
    source_score = [] #[seed,depth,source]
    source_size = []

    for i in range(n_seeds):
        temp_accuracy_score, rf, temp_source_score, temp_source_size = forest_wrapper(
            n_estimators = n_estimators, 
            depth = depth, dataset = dataset, 
            parameters=parameters, 
            metric = metric,
            seed = seed + i*10,
            plot = False)
        
        forests.append(rf)
        accuracy_score.append(temp_accuracy_score)
        source_score.append(temp_source_score)
        source_size.append(temp_source_size)
        #print(f"Done with Forest {i}")
    
    if(plot == True):
        accuracy_mean = np.mean(accuracy_score, axis=0)  # shape: (n_depths,)
        accuracy_std  = np.std(accuracy_score, axis=0)   # shape: (n_depths,)

        
        depth_range = np.arange(depth)
        plt.errorbar(depth_range, accuracy_mean, yerr = accuracy_std)
        plt.xlabel('Label Depth')
        plt.ylabel('Accuracy (R^2)')
        plt.xticks(np.arange(0,depth,step = 1))

    return accuracy_score, forests, source_score, source_size

def errorbars_stochastic_data(n_seeds, n_estimators, data, depth, parameters, metric="score", seed = 24, plot = True):
    forests = [] # forest[0] is forests with same dataset
    accuracy_score = []
    source_score = [] #[seed,depth,source]
    source_size = []

    X_tests, y_tests = [], []

    '''
    Data is only X, y. Each seed should change the dataset sampling
    '''

    for i in range(n_seeds):
        X_train, X_test, y_train, y_test = split_by_source(data.X, data.y, test_size=0.2, random_state=seed + 10*i)

        X_tests.append(X_test)
        y_tests.append(y_test)


        dataset = Dataset(X_train, X_test, y_train, y_test)

        temp_accuracy_score, rf, temp_source_score, temp_source_size = forest_wrapper(
            n_estimators = n_estimators, 
            depth = depth, 
            dataset = dataset, 
            parameters=parameters, 
            metric = metric,
            seed = seed,
            plot = False)
        
        forests.append(rf)
        accuracy_score.append(temp_accuracy_score)
        source_score.append(temp_source_score)
        source_size.append(temp_source_size)
        #print(f"Done with Forest {i}")
    
    if(plot == True):
        accuracy_mean = np.mean(accuracy_score, axis=0)
        accuracy_std  = np.std(accuracy_score, axis=0)

        
        depth_range = np.arange(depth)
        plt.errorbar(depth_range, accuracy_mean, yerr = accuracy_std)
        plt.xlabel('Label Depth')
        plt.ylabel('Accuracy (R^2)')
        plt.xticks(np.arange(0,depth,step = 1))

    return accuracy_score, forests, source_score, source_size, X_tests, y_tests

def errorbars_kfold_data(n_folds, n_estimators, depth, data, parameters, metric="score", seed = 24, plot = True):
    forests = [] # forest[0] is forests with same dataset
    accuracy_score = []
    source_score = [] #[seed,depth,source]
    source_size = []

    X_tests, y_tests = [], []

    '''
    Data is only X, y. Each seed should change the dataset sampling
    '''

    unique_sources = np.unique(data.X[:, -1])

    # build per-source KFold indices
    source_folds = {}
    for source in unique_sources:
        mask = np.where(data.X[:, -1] == source)[0]
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        source_folds[source] = list(kf.split(mask))  # list of (train_idx, test_idx) into mask


    for i in range(n_folds):
        X_train_parts, X_test_parts = [], []
        y_train_parts, y_test_parts = [], []

        for source in unique_sources:
            mask = np.where(data.X[:, -1] == source)[0]
            train_idx, test_idx = source_folds[source][i]

            X_train_parts.append(data.X[mask[train_idx]])
            X_test_parts.append(data.X[mask[test_idx]])
            y_train_parts.append(data.y[mask[train_idx]])
            y_test_parts.append(data.y[mask[test_idx]])

        X_train = np.concatenate(X_train_parts, axis=0)
        X_test  = np.concatenate(X_test_parts,  axis=0)
        y_train = np.concatenate(y_train_parts, axis=0)
        y_test  = np.concatenate(y_test_parts,  axis=0)

        X_tests.append(X_test)
        y_tests.append(y_test)


        dataset = Dataset(X_train, X_test, y_train, y_test)

        temp_accuracy_score, rf, temp_source_score, temp_source_size = forest_wrapper(
            n_estimators = n_estimators, 
            depth = depth, 
            dataset = dataset, 
            parameters=parameters, 
            metric = metric,
            seed = seed,
            plot = False)
        
        forests.append(rf)
        accuracy_score.append(temp_accuracy_score)
        source_score.append(temp_source_score)
        source_size.append(temp_source_size)
        #print(f"Done with Forest {i}")
    
    if(plot == True):
        accuracy_mean = np.mean(accuracy_score, axis=0)
        accuracy_std  = np.std(accuracy_score, axis=0)

        
        depth_range = np.arange(depth)
        plt.errorbar(depth_range, accuracy_mean, yerr = accuracy_std)
        plt.xlabel('Label Depth')
        plt.ylabel('Accuracy (R^2)')
        plt.xticks(np.arange(0,depth,step = 1))

    return accuracy_score, forests, source_score, source_size, X_tests, y_tests

def msl_errorbars_kfold_data(n_folds, n_estimators, n_forests, data, parameters, metric="score", seed = 24, plot = True):
    forests = [] # forest[0] is forests with same dataset
    accuracy_score = []
    source_score = [] #[seed,depth,source]
    source_size = []

    X_tests, y_tests = [], []

    '''
    Data is only X, y. Each seed should change the dataset sampling
    '''

    unique_sources = np.unique(data.X[:, -1])

    # build per-source KFold indices
    source_folds = {}
    for source in unique_sources:
        mask = np.where(data.X[:, -1] == source)[0]
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        source_folds[source] = list(kf.split(mask))  # list of (train_idx, test_idx) into mask


    for i in range(n_folds):
        X_train_parts, X_test_parts = [], []
        y_train_parts, y_test_parts = [], []

        for source in unique_sources:
            mask = np.where(data.X[:, -1] == source)[0]
            train_idx, test_idx = source_folds[source][i]

            X_train_parts.append(data.X[mask[train_idx]])
            X_test_parts.append(data.X[mask[test_idx]])
            y_train_parts.append(data.y[mask[train_idx]])
            y_test_parts.append(data.y[mask[test_idx]])

        X_train = np.concatenate(X_train_parts, axis=0)
        X_test  = np.concatenate(X_test_parts,  axis=0)
        y_train = np.concatenate(y_train_parts, axis=0)
        y_test  = np.concatenate(y_test_parts,  axis=0)

        X_tests.append(X_test)
        y_tests.append(y_test)


        dataset = Dataset(X_train, X_test, y_train, y_test)

        temp_accuracy_score, rf, temp_source_score, temp_source_size = msl_forest_wrapper(
            n_estimators = n_estimators,
            n_forests = n_forests, 
            dataset = dataset, 
            parameters=parameters, 
            metric = metric,
            seed = seed,
            plot = False)
        
        forests.append(rf)
        accuracy_score.append(temp_accuracy_score)
        source_score.append(temp_source_score)
        source_size.append(temp_source_size)
        #print(f"Done with Forest {i}")
    
    msl_set = np.linspace(len(dataset.X_train), parameters.min_samples_leaf, num=n_forests, dtype=int)

    if(plot == True):
        accuracy_mean = np.mean(accuracy_score, axis=0)
        accuracy_std  = np.std(accuracy_score, axis=0)

        plt.errorbar(msl_set, accuracy_mean, yerr = accuracy_std)
        plt.xlabel('Label Unmask Minimum Samples')
        plt.ylabel('Accuracy')
        plt.xticks(msl_set)
        plt.gca().invert_xaxis()

    return accuracy_score, forests, source_score, source_size, X_tests, y_tests

def split_by_source(X, y, test_size=0.2, random_state=24):
    unique_sources = np.unique(X[:, -1])
    
    X_train_parts, X_test_parts = [], []
    y_train_parts, y_test_parts = [], []
    
    for source in unique_sources:
        mask = X[:, -1] == source
        X_src = X[mask]
        y_src = y[mask]
        
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_src, y_src, 
            test_size=test_size, 
            random_state=random_state
        )
        
        X_train_parts.append(X_tr)
        X_test_parts.append(X_te)
        y_train_parts.append(y_tr)
        y_test_parts.append(y_te)
    
    X_train = np.concatenate(X_train_parts, axis=0)
    X_test  = np.concatenate(X_test_parts,  axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    y_test  = np.concatenate(y_test_parts,  axis=0)
    
    return X_train, X_test, y_train, y_test

def plot_forests(accuracy_score, source_score, x_values=None, x_label = "Label Unmask Depth", sourced=False, show_error = True):

    # default to integer index range if no x_values given
    
    if not sourced:

        accuracy_mean = np.mean(accuracy_score, axis=0)
        accuracy_std  = np.std(accuracy_score,  axis=0)

        if show_error == True:
            yerr = accuracy_std
        else:
            yerr = None

        if x_values is None:
            x_values = np.arange(len(accuracy_mean))
        else:
            x_values = np.asarray(x_values)

        plt.errorbar(x_values, accuracy_mean, yerr=yerr)
        plt.xlabel(x_label)
        plt.xticks(x_values)

    else:
        sourced_mean = np.mean(source_score, axis=0)
        sourced_std  = np.std(source_score,  axis=0)

        accuracy_mean = np.mean(accuracy_score, axis=0)
        accuracy_std  = np.std(accuracy_score,  axis=0)

        if show_error == True:
            yerr_avg = accuracy_std
        else:
            yerr_avg = None

        if x_values is None:
            x_values = np.arange(len(sourced_mean))
        else:
            x_values = np.asarray(x_values)

        for i in range(np.array(source_score).shape[2]):
            if show_error == True:
                yerr_source = sourced_std[:, i]
            else:
                yerr_source = None
            plt.errorbar(x_values, sourced_mean[:, i], yerr=yerr_source, label=f"Source {i}", alpha = 0.85)


        plt.errorbar(x_values, accuracy_mean, yerr=yerr_avg, linestyle='dashed', color = 'black', label = "All Data")


        plt.xlabel('Label Depth')
        plt.gca().invert_xaxis()
        plt.xticks(x_values)
        plt.legend()

def plot_forests_by_fold(accuracy_score, dataset, parameters, depth_type = "depth"):
    if depth_type == "depth":
        accuracy_score = np.asarray(accuracy_score)  # (n_folds, n_depths)
        n_folds, n_depths = accuracy_score.shape
        x_values = np.arange(n_depths)
        accuracy_mean = accuracy_score.mean(axis=0)

        n_rows = 2
        n_cols = int(np.ceil(n_folds / n_rows))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows), sharey=True)
        axes = axes.flatten()

        for fold_idx in range(n_folds):
            ax = axes[fold_idx]
            ax.plot(x_values, accuracy_score[fold_idx], marker='o', alpha=0.8, color='C0')
            ax.plot(x_values, accuracy_mean, linestyle='dashed', color='black', alpha=0.6)
            ax.set_title(f"Fold {fold_idx}")
            ax.set_xticks(x_values)

        # hide any unused subplot slots
        for j in range(n_folds, len(axes)):
            axes[j].axis('off')

        fig.supxlabel("Label Unmask Depth")
        fig.supylabel("Accuracy")
        plt.tight_layout()
    elif depth_type == "msl":
        accuracy_score = np.asarray(accuracy_score)  # (n_folds, n_forests)
        n_folds, n_forests = accuracy_score.shape

        # msl_set: same array used when the forests were generated, e.g.
        msl_set = np.linspace(len(dataset.X_train), parameters.min_samples_leaf, num=n_forests, dtype=int)
        x_values = msl_set
        accuracy_mean = accuracy_score.mean(axis=0)

        n_rows = 2
        n_cols = int(np.ceil(n_folds / n_rows))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows), sharey=True)
        axes = axes.flatten()

        for fold_idx in range(n_folds):
            ax = axes[fold_idx]
            ax.plot(x_values, accuracy_score[fold_idx], marker='o', alpha=0.8, color='C0')
            ax.plot(x_values, accuracy_mean, linestyle='dashed', color='black', alpha=0.6)
            ax.set_title(f"Fold {fold_idx}")
            ax.set_xticks(x_values)
            ax.invert_xaxis()  # larger min_samples_label = less unmasking, so flip like your original msl plots

        for j in range(n_folds, len(axes)):
            axes[j].axis('off')

        fig.supxlabel("Min Samples Label")
        fig.supylabel("Accuracy")
        plt.tight_layout()
    else:
        print("incorrect depth type")

def evaluate_forests(forests, X_tests, y_tests, metric="score"):
    '''
    forests:  [seed][depth] = fitted RandomForestClassifier621
    X_tests:  [seed] = X_test for that seed (list if data varies per seed, 
               or a single array if same split for all seeds)
    y_tests:  [seed] = y_test for that seed
    '''
    n_seeds  = len(forests)
    n_depths = len(forests[0])

    # normalize single array input to per-seed lists
    # so this works for both errorbars_stochastic_forest (one fixed split)
    # and errorbars_stochastic_data (different split per seed)
    if isinstance(X_tests, np.ndarray):
        X_tests = [X_tests] * n_seeds
        y_tests = [y_tests] * n_seeds

    if metric not in ("score", "auc"):
        raise ValueError(f"Unknown metric: {metric!r}. Expected 'score' or 'auc'.")

    metric_fn = lambda model, X, y: model.auc_score(X, y) if metric == "auc" else model.score(X, y)

    # use first seed's test set to determine n_sources (consistent across seeds by design)
    unique_vals = np.unique(X_tests[0][:, -1])
    n_sources = len(unique_vals)

    accuracy_score = np.empty((n_seeds, n_depths))
    source_score = np.empty((n_seeds, n_depths, n_sources))
    source_size = []

    for s in range(n_seeds):
        X_test = X_tests[s]
        y_test = y_tests[s]

        for d in range(n_depths):
            accuracy_score[s][d] = metric_fn(forests[s][d], X_test, y_test)

            for i, source in enumerate(unique_vals):
                sourced_X = X_test[X_test[:, -1] == source]
                sourced_y = y_test[X_test[:, -1] == source]
                source_score[s][d][i] = metric_fn(forests[s][d], sourced_X, sourced_y)

    for source in unique_vals:
        source_size.append(np.sum(X_tests[0][:, -1] == source))

    return accuracy_score, source_score, source_size
