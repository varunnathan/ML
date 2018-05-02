import warnings

import pandas as pd
import sys
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter


def _calc_outlier_bounds(numbers):
    """Return low and high bounds that excludes outliers."""

    l, h = np.percentile(numbers, [25, 75])
    r = (h - l) * 1.5
    return (max(l - r, np.min(numbers)),
            min(h + r, np.max(numbers)))


class MaxAUCImputerBase(BaseEstimator, TransformerMixin):

    def transform(self, X):
        """Impute all missing values in X.

        Parameters
        ----------
        X : {array-like}, shape = (n_samples, n_features)
            The input data to complete.
        Return
        ------
        X_new : {array-like},
                Transformed array.
                shape (n_samples, n_features).
        """

        X = check_array(X, copy=True, ensure_2d=True, force_all_finite=False)
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            if mask.any():
                X[mask, i] = self.nan_value[i]
        return X


class MaxAUCImputerCV(MaxAUCImputerBase):
    """Imputation transformer for completing missing values.

    Impute missing values identified using `np.isnan` with
    the following that results in highest AUC:

      * `np.percentile(x, 50)` (median),
      * `np.percentile(x, 0)` (min),
      * `np.percentile(x, 100)` (max).
      * `np.percentile(x, 25)`,
      * `np.percentile(x, 75)`.
      * `x.mean())`,
      * zero (`0`).

    AUC is calculated through cross validation.

    When all attemps result in same AUC, first one, median, is used.


    Fitter imputer has following attributes that are list of the same length
    as number of features:

      * `strategy`: imputation strategy per feature
      * `nan_value`: value that replaced NaNs
      * `auc_cv`: maximum AUC value obtained in cross-validation
      * `auc_max`: maximum AUC value obtained after imputation
      * `auc_median`: AUC value obtained after median imputation for
        comparison purposes
    """
    def __init__(self, verbose=0, copy=True, cv_folds=2):
        self.verbose = verbose
        self.copy = copy    # not used
        self.cv_folds = cv_folds

    def fit(self, X, y):
        """Fit the imputer on X.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        Returns
        -------
        self : object
            Returns self.
        """
        from sklearn.metrics import roc_auc_score
        from sklearn.cross_validation import StratifiedKFold

        folds = []
        if self.cv_folds:
            for train, test in StratifiedKFold(y, n_folds=self.cv_folds,
                                               random_state=20160501):
                if len(set(y[test])) == 1:
                    continue
                folds.append((train, test))

        if len(folds) <= 1:
            folds = [(None, None)]

        strategy = [
            "median",
            "min",
            "max",
            "p25",
            "p75",
            "mean",
            "zero"]

        median_index = strategy.index('median')
        percentiles = [50, 0, 100, 25, 75]

        self.strategy = []
        self.nan_value = []
        self.auc_cv = []
        self.auc_max = []
        self.auc_median = []

        def calc_auc(y, x):
            """Return maximum of AUC or 1-AUC."""
            auc = roc_auc_score(y, x)
            return max(auc, 1 - auc)

        def get_values(x):
            """Return values to replace NaNs and filling strategy names."""
            x = x[~np.isnan(x)]
            values = (list(np.percentile(x, percentiles)) +
                      [x.mean(), 0])
            return values

        def get_aucs(x, y, values):
            """Return AUCs calculated after NaNs are filled with values."""
            nans = np.isnan(x)
            aucs = []
            for value in values:
                x[nans] = value
                aucs.append(calc_auc(y, x))
            return aucs

        for i in range(X.shape[1]):
            x = X[:, i].copy()
            auc_folds = []
            for train, test in folds:
                if train is None:
                    x_train = x
                    x_test = x
                    y_test = y
                else:
                    x_train = x[train]
                    x_test = x[test]
                    y_test = y[test]

                if (sum(np.isnan(x_train)) == len(x_train) or
                        sum(np.isnan(x_test)) == len(x_test)):
                    x_train = x
                    x_test = x
                    y_test = y

                values = get_values(x_train)
                aucs = get_aucs(x_test, y_test, values)
                auc_folds.append(aucs)

            auc_folds = np.mean(auc_folds, 0)
            index = np.argmax(auc_folds)

            values = get_values(x)
            aucs = get_aucs(x, y, values)

            self.strategy.append(strategy[index])
            self.nan_value.append(values[index])
            self.auc_cv.append(auc_folds[index])
            self.auc_max.append(auc_folds[index])
            self.auc_median.append(aucs[median_index])
        return self


class LegacyOutlierScaler(BaseEstimator, TransformerMixin):

    def __init__(self, copy=True):

        self.copy = copy

    def __repr__(self):

        return '{}(copy={})'.format(self.__class__.__name__, repr(self.copy))

    def fit(self, X, y=None):

        self.low = []
        self.high = []
        self.low_default = []
        self.high_default = []
        self.low_scaler = []
        self.high_scaler = []

        for j in range(X.shape[1]):
            x = X[:, j].copy()
            low, high = _calc_outlier_bounds(x)
            self.low.append(low)
            self.high.append(high)
            if low == high:
                self.low_default.append(None)
                self.low_scaler.append(None)
                self.high_scaler.append(None)
                self.high_default.append(None)
                continue
            q1, q3 = np.percentile(x, [25, 75])
            qr = (q3 - q1) / 2.
            qrd = qr / 50

            min_, max_ = x.min(), x.max()

            if min_ < low:
                self.low_scaler.append(min(qr / (low - min_), 1))
                self.low_default.append(None)
            else:
                self.low_scaler.append(None)
                self.low_default.append(low - qrd)

            if max_ > high:
                self.high_scaler.append(min(qr / (max_ - high), 1))
                self.high_default.append(None)
            else:
                self.high_scaler.append(None)
                self.high_default.append(high + qrd)
        return self

    def transform(self, X, y=None, copy=None):

        if self.copy:
            X_ = X.copy()
        else:
            X_ = X
        for j in range(X.shape[1]):
            inter = False
            low = self.low[j]
            mask = X_[:, j] < low
            if any(mask):
                default = self.low_default[j]
                scaler = self.low_scaler[j]
                if default is not None:
                    X_[mask, j] = default
                    inter = True
                elif scaler is not None:
                    X_[mask, j] = low - (low - X_[mask, j]) * self.low_scaler[j]
                    inter = True

            high = self.high[j]
            mask = X_[:, j] > high
            if any(mask):
                default = self.high_default[j]
                scaler = self.high_scaler[j]
                if default is not None:
                    X_[mask, j] = default
                elif scaler is not None:
                    X_[mask, j] = high + (X_[mask, j] - high)*self.high_scaler[j]
            if 0 and inter:
                x = X[:, j]
                x_ = X_[:, j]
                from code import interact; interact(local=locals())
        return X_


def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)

    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)


def _ensure_sparse_format(spmatrix, accept_sparse, dtype, order, copy,
                          force_all_finite):
    """Convert a sparse matrix to a given format.

    Checks the sparse format of spmatrix and converts if necessary.

    Parameters
    ----------
    spmatrix : scipy sparse matrix
        Input to validate and convert.

    accept_sparse : string, list of string or None (default=None)
        String[s] representing allowed sparse matrix formats ('csc',
        'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). None means that sparse
        matrix input will raise an error.  If the input is sparse but not in
        the allowed format, it will be converted to the first listed format.

    dtype : string, type or None (default=none)
        Data type of result. If None, the dtype of the input is preserved.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.

    Returns
    -------
    spmatrix_converted : scipy sparse matrix.
        Matrix that is ensured to have an allowed type.
    """
    if accept_sparse is None:
        raise TypeError('A sparse matrix was passed, but dense '
                        'data is required. Use X.toarray() to '
                        'convert to a dense numpy array.')
    sparse_type = spmatrix.format
    if dtype is None:
        dtype = spmatrix.dtype
    if sparse_type in accept_sparse:
        # correct type
        if dtype == spmatrix.dtype:
            # correct dtype
            if copy:
                spmatrix = spmatrix.copy()
        else:
            # convert dtype
            spmatrix = spmatrix.astype(dtype)
    else:
        # create new
        spmatrix = spmatrix.asformat(accept_sparse[0]).astype(dtype)
    if force_all_finite:
        if not hasattr(spmatrix, "data"):
            warnings.warn("Can't check %s sparse matrix for nan or inf."
                          % spmatrix.format)
        else:
            _assert_all_finite(spmatrix.data)
    if hasattr(spmatrix, "data"):
        spmatrix.data = np.array(spmatrix.data, copy=False, order=order)
    return spmatrix


def check_array(array, accept_sparse=None, dtype=None, order=None, copy=False,
                force_all_finite=True, ensure_2d=True, allow_nd=False):
    """Input validation on an array, list, sparse matrix or similar.

    By default, the input is converted to an at least 2nd numpy array.

    Parameters
    ----------
    array : object
        Input  to check / convert.

    accept_sparse : string, list of string or None (default=None)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc.  None means that sparse matrix input will raise an error.
        If the input is sparse but not in the allowed format, it will be
        converted to the first listed format.

    dtype : string, type or None (default=none)
        Data type of result. If None, the dtype of the input is preserved.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.

    ensure_2d : boolean (default=True)
        Whether to make X at least 2d.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    Returns
    -------
    X_converted : object
        The converted and validated X.
    """
    if isinstance(accept_sparse, str):
        accept_sparse = [accept_sparse]

    if sp.issparse(array):
        array = _ensure_sparse_format(array, accept_sparse, dtype, order,
                                      copy, force_all_finite)
    else:
        if ensure_2d:
            array = np.atleast_2d(array)
        array = np.array(array, dtype=dtype, order=order, copy=copy)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. Expected <= 2" %
                             array.ndim)
        if force_all_finite:
            _assert_all_finite(array)

    return array


def _convert_to_float(df):
    try:
        values = df.values.astype(float)
    except ValueError:
        for col in df:
            try:
                df[col].astype(float)
            except ValueError:
                exit('{} could not be converted to float, '
                    'values it takes are: {}'.format(col, dict(Counter(df[col]))))
    else:
        return values


def rfe_feats(df, feat_names=None, sample_col=None, target_col='DV32',
              thresh=0.1, step=0.05, verbose=False, **kwargs):
    """ Select features using recursive-feature elimination,
        model used is LogisticRegression, inputs are ~
            df: Input pandas dataframe with features and target_col
            feat_names: list of features to consider
            sample_col: Specifies dev/val splits
            target_col: Dependent variable
            thresh: Fraction of feats from all specified input features
            step: Fraction of feats to eliminate at each step
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import RFE
    from sklearn.preprocessing import Imputer

    from sys import exit
    from math import floor

    for col, val in [('target_col', target_col)]:
        if val not in df:
            exit('{}={} is not in data'.format(col, repr(val)))

    if sample_col:
        df = df[df[sample_col] == 'dev'].copy()

    model = LogisticRegression()

    if not feat_names:
        feat_names = [col for col in df if col not in (target_col, sample_col)]

    X = _convert_to_float(df[feat_names])
    y = _convert_to_float(df[target_col])
    if verbose:
        sys.stderr.write('initial shape of df: {}\n'.format(X.shape))

    infs = np.isinf(X)
    if infs.any():
        X[infs] = np.nan
    all_nans = np.isnan(X).sum(0) == len(X)
    if verbose and all_nans.any():
        sys.stderr.write('Dropping following columns as all have NaNs in dev '
                         'sample: {}\n'
                         .format(', '.join([f for i, f in enumerate(feat_names)
                                            if all_nans[i]])))
    X = X[:, ~all_nans]
    nonnullcols = [f for i, f in enumerate(feat_names) if not all_nans[i]]
    if verbose:
        sys.stderr.write('after dropping columns: {}\n'.format(X.shape))

    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    imp_fit = imp.fit(X)
    X_imp = imp_fit.transform(X)
    rfe = RFE(model, floor(len(feat_names)*thresh), verbose=0, step=step)
    rfe = rfe.fit(X_imp, y)
    if verbose:
        sys.stderr.write('# of selected feats: {}\n'.format(rfe.support_.shape))

    selected_feats = list(pd.Series(nonnullcols)[rfe.support_].values)
    return selected_feats


def dfsort(df):

    return getattr(df, 'sort_values', getattr(df, 'sort'))


def mutinfo_table(data, target):
    """Calculate mutual information and quantile table.

    *data* must be a DataFrame that contains independent variables,
    and optionally *target* variable. If *target* is contained in data,
    *target* must be a column name. Otherwise, it must be a series or
    an array that aligns with *data*.

    Output table contains:

      * `Mutinfo`: mutual information
    """

    if hasattr(target, 'upper'):
        try:
            tcol = target
            target = data[target].values
        except KeyError:
            raise KeyError('{} is not a column in data'.format(target))
        else:
            cols = [col for col in data if col != tcol]

    else:
        cols = list(data)
        try:
            target = target.values
        except AttributeError:
            pass

    labels = ['MutInfo', 'Q1Rate', 'Q2Rate', 'Q3Rate', 'Q4Rate',
              'NAFrac', 'NACount', 'NARate']

    result = []
    for col in cols:
        feature = data[col]

        mi = calc_mutinfo(feature, target)
        result.append([mi] + _calc_true_rates(feature, target, 4))

    ff = pd.DataFrame(np.array(result), columns=labels, index=cols)
    ff['MutInfo'] = ff['MutInfo'].round(3)
    for col in ff:
        if 'Rate' in col:
            ff[col] = ff[col].round(3)
    ff['NAFrac'] = ff['NAFrac'].round(3)
    dfsort(ff)('MutInfo', ascending=False, inplace=1)
    ff['MutInfo'] = ff.MutInfo.round(4)
    return ff


def calc_mutinfo(feature, target):
    """Calculate mutual information after binning *feature* with separate
    bins for missing values, low and high end outliers."""

    if hasattr(target, 'values'):
        target = target.values

    binned = feature.copy()
    binned[:] = -2
    binned[feature.isnull()] = -1
    if feature.nunique() > 10:
        low, high = _calc_outlier_bounds(feature[feature.notnull()].values)
        binned[feature < low] = 0
        ibin = 1
        bounds = np.linspace(low, high, 11)
        for low, high in zip(bounds[:-1], bounds[1:]):
            mask = (feature >= low) & (feature < high)
            if mask.any():
                binned[mask] = ibin
                ibin += 1
        binned[feature >= high] = ibin
    else:
        ibin = 1
        for val in feature.unique():
            binned[feature == val] = ibin
            ibin += 1

    from sklearn.metrics import mutual_info_score
    return mutual_info_score(binned.values, target)


def _calc_true_rates(feature, target, nbins=4):
    """Calculate True rate in bins split after target is sorted according to
    feature. When there are null values, they are omitted from binning
    and True rate is calculated for null values. Also, percentage of
    null values is reported."""

    if hasattr(feature, 'isnull'):
        mask = feature.isnull()
        if mask.any():
            mask = mask.values
            null_pct = mask.mean()
            null_cnt = mask.sum()
            null_rate = target[mask].mean()
            mask = feature.notnull().values
            feature = feature[mask]
            target = target[mask]
        else:
            null_rate = np.nan
            null_pct = 0.
            null_cnt = 0
    else:
        null_rate = np.nan
        null_pct = np.nan
        null_cnt = 0

    rates = []
    zipped = list(zip(feature, target))
    zipped.sort()
    arr = np.array(zipped)[:, 1]
    ls = np.linspace(0, len(arr), nbins+1).round().astype(int)
    for low, high in zip(ls[:-1], ls[1:]):
        rates.append(arr[low:high].mean())

    rates.append(null_pct)
    rates.append(null_cnt)
    rates.append(null_rate)
    return rates


def cluster_features(df, target_col, corrcoef_cutoff=0.7, scores=None,
                     drop_uniform=True, **kwargs):

    target_name = None
    if hasattr(target_col, 'upper'):
        bak = df
        try:
            target_name = target_col
            target_series = df[target_name]
            target_col = target_series.values
        except KeyError:
            raise KeyError('{} is not a column in data'.format(target_name))
        else:
            cols = [col for col in df if col != target_name]
            df = df[cols]

    if drop_uniform:
        df = df[[col for col in df if df[col].nunique() > 1]]

    feat_mutinfo = mutinfo_table(df, target_col)
    if scores is None:
        scores = feat_mutinfo.MutInfo
    feat_clusts = _cluster_features(df, df.columns,
                                   corrcoef_cutoff, scores)

    features = feat_clusts.join(feat_mutinfo)
    mean_mi = []
    for f, row in features.iterrows():
        if row.Size > 1:
            members = [m.strip() for m in row.Members.split(',')]
            mean_mi.append(feat_mutinfo.loc[members].MutInfo.mean())
        else:
            mean_mi.append(row.MutInfo)

    features['MeanMutInfo'] = np.array(mean_mi).round(4)
    if target_name:
        dv = bak[target_name].astype(float)
        for f in features.index:
            features.loc[f, 'TargetCorr'] = dv.corr(bak[f].astype(float))
    features = features[['TargetCorr',  'MutInfo', 'Size', 'MeanMutInfo',
        'Q1Rate', 'Q2Rate', 'Q3Rate', 'Q4Rate', 'NAFrac', 'NARate',
        'Members', 'Mean Correlation', 'Max Correlation', 'Min Correlation']]
    dfsort(features)('MutInfo', ascending=False, inplace=1)
    for col in features:
        if 'Corr' in col:
            features[col] = features[col].round(2)

    if kwargs.get('only_names', False):
        return features.index
    else:
        return features


def _cluster_features(data, column_names=None, cutoff=0.6, scores=None):
    """Cluster features based on correlations between them. Distance between
    features f1 and f2 is `1 - corrcoef(f1, f2).

    Each row of *data* must correspond to a feature.

    Features can be selected based on given *scores*, which must be
    a series indexed by feature *names*. If *scores* is not provided,
    feature that correlates most with others in the cluster will be
    selected.
    """

    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, fcluster

    if hasattr(data, 'fillna'):
        data = data.copy()
        for col in data:
            series = data[col]
            mask = series.isnull()
            if mask.any():
                data[col].fillna(series.mean(), inplace=1)

        if column_names is None:
            column_names = list(data.columns)
        data = data.astype(float).values.T

    if column_names is None:
        column_names = [str(i) for i in range(len(data))]

    cc = np.corrcoef(data)
    if np.isnan(cc).sum():
        raise ValueError('feature correlation matrix has NaNs')

    ccabs = np.abs(cc)
    dist = 1.0 - ccabs
    dist[dist <= 0] = 0
    np.fill_diagonal(dist, 0)

    vect = squareform(dist, checks=False)  # checks=F added for py3 compat.
    Z = linkage(vect)
    clusters_ = fcluster(Z, 1 - cutoff, criterion='distance')

    clust_ids = set(clusters_)
    n_clusters = len(clust_ids)

    index = []
    columns = ['Size', 'Mean Correlation',
               'Max Correlation', 'Min Correlation',
               'Members']

    clusters = []
    numbers = []
    clustreps = {}
    for c in set(clusters_):
        which = clusters_ == c
        nz = which.nonzero()[0]
        if not len(nz):
            continue
        members = [column_names[f] for f in nz]

        if which.sum() == 1:
            index.append(members[0])
            clusters.append((1, 1, 1, 1, members[0]))
        else:

            ccrows = ccabs[which]
            ccsub = ccrows[:, which]
            np.fill_diagonal(ccsub, 0)

            if scores is None:
                index.append(members[ccsub.sum(0).argmax()])
            else:
                ss = scores[members]
                index.append(ss.argmax())

            ccmax = ccsub.max(0)
            clusters.append((len(members), ccmax.mean(), ccmax.max(),
                            ccmax.min(), u', '.join(members)))

    df = pd.DataFrame(clusters, index=index, columns=columns)
    dfsort(df)('Size', inplace=1, ascending=False)
    return df
