from __future__ import absolute_import
import sklearn.base
import numpy as np

import vowpal_porpoise

# if you try to say "import sklearn", it thinks sklearn is this file :(


class _VW(sklearn.base.BaseEstimator):

    """scikit-learn interface for Vowpal Wabbit

    Only works for regression and binary classification.
    """

    def __init__(self,
                 logger=None,
                 vw='vw',
                 moniker='moniker',
                 name=None,
                 bits=None,
                 loss=None,
                 passes=10,
                 log_stderr_to_file=False,
                 silent=False,
                 l1=None,
                 l2=None,
                 learning_rate=None,
                 quadratic=None,
                 audit=None,
                 power_t=None,
                 adaptive=False,
                 working_dir=None,
                 decay_learning_rate=None,
                 initial_t=None,
                 minibatch=None,
                 total=None,
                 node=None,
                 unique_id=None,
                 span_server=None,
                 bfgs=None,
                 oaa=None,
                 old_model=None,
                 incremental=False,
                 mem=None,
                 nn=None,
                 ngrams=None,
                 skips=None,
                 classify=True
                 ):
        self.logger = logger
        self.vw = vw
        self.moniker = moniker
        self.name = name
        self.bits = bits
        self.loss = loss
        self.passes = passes
        self.log_stderr_to_file = log_stderr_to_file
        self.silent = silent
        self.l1 = l1
        self.l2 = l2
        self.learning_rate = learning_rate
        self.quadratic = quadratic
        self.audit = audit
        self.power_t = power_t
        self.adaptive = adaptive
        self.working_dir = working_dir
        self.decay_learning_rate = decay_learning_rate
        self.initial_t = initial_t
        self.minibatch = minibatch
        self.total = total
        self.node = node
        self.unique_id = unique_id
        self.span_server = span_server
        self.bfgs = bfgs
        self.oaa = oaa
        self.old_model = old_model
        self.incremental = incremental
        self.mem = mem
        self.nn = nn
        self.ngrams = ngrams
        self.skips = skips
        self.classify = classify

    def fit(self, X, y):
        """Fit Vowpal Wabbit

        Parameters
        ----------
        X: [{<feature name>: <feature value>}]
            input features
        y: [int or float]
            output labels
        """
        examples = _as_vw_strings(X, y)

        # initialize model
        self.vw_ = vowpal_porpoise.VW(
            logger=self.logger,
            vw=self.vw,
            moniker=self.moniker,
            name=self.name,
            bits=self.bits,
            loss=self.loss,
            passes=self.passes,
            log_stderr_to_file=self.log_stderr_to_file,
            silent=self.silent,
            l1=self.l1,
            l2=self.l2,
            learning_rate=self.learning_rate,
            quadratic=self.quadratic,
            audit=self.audit,
            power_t=self.power_t,
            adaptive=self.adaptive,
            working_dir=self.working_dir,
            decay_learning_rate=self.decay_learning_rate,
            initial_t=self.initial_t,
            minibatch=self.minibatch,
            total=self.total,
            node=self.node,
            unique_id=self.unique_id,
            span_server=self.span_server,
            bfgs=self.bfgs,
            oaa=self.oaa,
            old_model=self.old_model,
            incremental=self.incremental,
            mem=self.mem,
            nn=self.nn,
            ngrams=self.ngrams,
            skips=self.skips
        )

        # add examples to model
        with self.vw_.training():
            for instance in examples:
                self.vw_.push_instance(instance)

        # learning done after "with" statement
        return self

    def predict(self, X):
        """Fit Vowpal Wabbit

        Parameters
        ----------
        X: [{<feature name>: <feature value>}] or
           [<text features>] or [[<text features>]]
            input features
        """
        examples = _as_vw_strings(X)

        # add test examples to model
        with self.vw_.predicting():
            for instance in examples:
                self.vw_.push_instance(instance)

        # read out predictions
        predictions = np.asarray(list(self.vw_.read_predictions_()))

        return [1 if x >= 0 else -1 for x in predictions] if self.classify else predictions

    def predict_proba(self, X):
        """Fit Vowpal Wabbit

        Parameters
        ----------
        X: [{<feature name>: <feature value>}] or
           [<text features>] or [[<text features>]]
            input features
        """
        examples = _as_vw_strings(X)

        # add test examples to model
        with self.vw_.predicting():
            for instance in examples:
           
        # read out predictions
        predictions = list(self.vw_.read_predictions_())
        probabilities = [1 / (1 + np.exp(-x)) for x in predictions]
        return np.asarray([[1 - x, x] for x in probabilities])

        return [1 if x >= 0 else -1 for x in predictions] if self.classify else predictions




class VW_Regressor(sklearn.base.RegressorMixin, _VW):
    pass


class VW_Classifier(sklearn.base.ClassifierMixin, _VW):

    def predict(self, X):
        result = super(VW_Classifier, self).predict(X)
        return result


def _as_vw_string(x, y=None):
    """Convert {feature: value} to something _VW understands

    Parameters
    ----------
    x : {<feature>: <value>} or <text features> or [<text features>]
    y : int or float
    """
    result = str(y)
    if isinstance(x, str):
        return result + " | " + x
    elif isinstance(x, list):
        return result + " | " + " ".join([str(z) for z in x])
    elif ininstance(x, numpy.ndarray):
        return result + " | " + " ".join([str(z) for z in x])
    elif isinstance(x, dict):
        return result + " | " + " ".join(["%s:%f" % (key, value) for (key, value) in x.items()])
    else:
        raise TypeError("unsupported type: %s" % type(x))


def _as_vw_strings(X, y=None):
    n_samples = np.shape(X)[0]
    if y is None:
        y = np.ones(n_samples)
    return [_as_vw_string(X[i], y[i]) for i in range(n_samples)]
