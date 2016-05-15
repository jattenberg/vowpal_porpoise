from contextlib import contextmanager
import os
import sys
import subprocess
import shlex
import tempfile
from vp_utils import safe_remove, VPLogger
import socket
import atexit
import os

class VW:
    def __init__(self,
                 logger=None,
                 vw='vw',
                 moniker=None,
                 name=None,
                 bits=None,
                 loss=None,
                 passes=None,
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
                 lda=None,
                 lda_D=None,
                 lda_rho=None,
                 lda_alpha=None,
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
                 cleanup=False,
                 ftrl=False,
                 ftrl_alpha=None,
                 ftrl_beta=None
                 **kwargs):
        assert moniker and passes

        if logger is None:
            self.log = VPLogger()
        else:
            self.log = logger

        self.node = node
        self.total = total
        self.unique_id = unique_id
        self.span_server = span_server
        if self.node is not None:
            assert self.total is not None
            assert self.unique_id is not None
            assert self.span_server is not None

        if name is None:
            self.handle = '%s' % moniker
        else:
            self.handle = '%s.%s' % (moniker, name)

        if self.node is not None:
            self.handle = "%s.%d" % (self.handle, self.node)

        if old_model is None:
            self.filename = '%s.model' % self.handle
            self.incremental = False
        else:
            self.filename = old_model
            self.incremental = True

        self.incremental = incremental
        self.filename = '%s.model' % self.handle

        self.name = name
        self.bits = bits
        self.loss = loss
        self.vw = vw
        self.l1 = l1
        self.l2 = l2
        self.learning_rate = learning_rate
        self.log_stderr_to_file = log_stderr_to_file
        self.silent = silent
        self.passes = passes
        self.quadratic = quadratic
        self.power_t = power_t
        self.adaptive = adaptive
        self.decay_learning_rate = decay_learning_rate
        self.audit = audit
        self.initial_t = initial_t
        self.lda = lda
        self.lda_D = lda_D
        self.lda_rho = lda_rho
        self.lda_alpha = lda_alpha
        self.minibatch = minibatch
        self.oaa = oaa
        self.bfgs = bfgs
        self.mem = mem
        self.nn = nn
        self.ngrams = ngrams
        self.skips = skips
        self.cleanup = cleanup
        self.ftrl = ftrl
        self.ftrl_alpha = ftrl_alpha
        self.ftrl_beta = ftrl_beta

        # Do some sanity checking for compatability between models
        if self.lda:
            assert not self.l1
            assert not self.l1
            assert not self.l2
            assert not self.loss
            assert not self.adaptive
            assert not self.oaa
            assert not self.bfgs
        else:
            assert not self.lda_D
            assert not self.lda_rho
            assert not self.lda_alpha
            assert not self.minibatch

        self.working_directory = working_dir or os.getcwd()

    def vw_base_command(self, base):
        l = base
        if self.bits                is not None: l.append('-b %d' % self.bits)
        if self.learning_rate       is not None: l.append('--learning_rate=%f' % self.learning_rate)
        if self.l1                  is not None: l.append('--l1=%f' % self.l1)
        if self.l2                  is not None: l.append('--l2=%f' % self.l2)
        if self.initial_t           is not None: l.append('--initial_t=%f' % self.initial_t)
        if self.quadratic           is not None: l.append('-q %s' % self.quadratic)
        if self.power_t             is not None: l.append('--power_t=%f' % self.power_t)
        if self.loss                is not None: l.append('--loss_function=%s' % self.loss)
        if self.decay_learning_rate is not None: l.append('--decay_learning_rate=%f' % self.decay_learning_rate)
        if self.lda                 is not None: l.append('--lda=%d' % self.lda)
        if self.lda_D               is not None: l.append('--lda_D=%d' % self.lda_D)
        if self.lda_rho             is not None: l.append('--lda_rho=%f' % self.lda_rho)
        if self.lda_alpha           is not None: l.append('--lda_alpha=%f' % self.lda_alpha)
        if self.minibatch           is not None: l.append('--minibatch=%d' % self.minibatch)
        if self.oaa                 is not None: l.append('--oaa=%d' % self.oaa)
        if self.unique_id           is not None: l.append('--unique_id=%d' % self.unique_id)
        if self.total               is not None: l.append('--total=%d' % self.total)
        if self.node                is not None: l.append('--node=%d' % self.node)
        if self.span_server         is not None: l.append('--span_server=%s' % self.span_server)
        if self.mem                 is not None: l.append('--mem=%d' % self.mem)
        if self.audit:                           l.append('--audit')
        if self.bfgs:                            l.append('--bfgs')
        if self.adaptive:                        l.append('--adaptive')
        if self.nn                  is not None: l.append('--nn=%d' % self.nn)
        if self.ngrams              is not None: l.append('--ngram %d' % self.ngrams)
        if self.skips               is not None: l.append('--skips %d' % self.skips)
        if self.ftrl:                            l.append('--ftrl')
        if self.ftrl_alpha          is not None: l.append('--ftrl_alpha %d' % self.ftrl_alpha)
        if self.ftrl_beta           is not None: l.append('--ftrl_beta %d' % self.ftrl_beta)
        return ' '.join(l)


    def vw_file_training_command(self, cache_file, model_file, training_file):
        if os.path.exists(model_file) and self.incremental:
            return self.vw_base_command([self.vw]) + ' --passes %d --cache_file %s -i %s -f %s -d %s' \
                    % (self.passes, cache_file, model_file, model_file, training_file)
        else:
            self.log.debug('No existing model file or not options.incremental')
            return self.vw_base_command([self.vw]) + ' --passes %d --cache_file %s -f %s -d %s' \
                    % (self.passes, cache_file, model_file, training_file)


    def vw_file_test_command(self, model_file, prediction_file, example_file):
        return self.vw_base_command([self.vw]) + ' -t -i %s -d %s -p %s' % (model_file, example_file, prediction_file)


    def cleanup(self):
        cache_file = self.get_cache_file()
        model_file = self.get_model_file()
        safe_remove(cache_file)
        safe_remove(model_file)
        safe_remove(self.training_file)
        safe_remove(self.prediction_file)

    @contextmanager
    def training(self):
        self.start_training()
        yield
        self.close_process()

    @contextmanager
    def predicting(self):
        self.start_predicting()
        yield
        self.close_process()

    def start_training(self):
        _, example_file = tempfile.mkstemp(dir='.', prefix=self.get_example_file("training"))
        os.close(_)
        self.training_handle = open(example_file, 'w')
        self.training_file = example_file
        
        self.log.debug("storing training instancs: %s for %s " % (example_file, self.handle) )

        # Remove the old cache and model files
        if not self.incremental:
            cache_file = self.get_cache_file()
            model_file = self.get_model_file()
            safe_remove(cache_file)
            safe_remove(model_file)


    def start_predicting(self):
        _, example_file = tempfile.mkstemp(dir='.', prefix=self.get_example_file("testing"))
        os.close(_)
        _, prediction_file = tempfile.mkstemp(dir='.', prefix=self.get_prediction_file())
        os.close(_)

        self.log.debug("storing test instancs: %s for %s" % (example_file, self.handle) )
        self.log.debug("writing predictions: %s for %s" % (prediction_file, self.handle) )

        self.training_handle = open(example_file, 'w')
        self.training_file = example_file
        self.read_predictions = self.read_predictions_
        self.prediction_file = prediction_file

    def close_process(self):
        self.training_handle.close()
        if self.cleanup:
            safe_remove(self.training_file)


    def train(self, examples):
        for example in examples:
            self.training_handle.write(example + "\n")

        self.training_handle.flush()
        model_file = self.get_model_file()
        cache_file = self.get_cache_file()
        self.log.debug("storing model: %s for %s" % (model_file, self.handle))

        command = self.vw_file_training_command(cache_file, model_file, self.training_file)
        self.log.debug(command)
        os.system(command)

    def predict(self, examples):
        for example in examples:
            self.training_handle.write(example + "\n")
        self.training_handle.flush()
        model_file = self.get_model_file()

        command = self.vw_file_test_command(model_file, self.prediction_file, self.training_file)
        self.log.debug(command)
        os.system(command)

    def parse_prediction(self, p):
        if self.lda:
            return map(float, p.split())
        else:
            return float(p.split()[0])

    def read_predictions_(self):
        for x in open(self.prediction_file):
            yield self.parse_prediction(x)
        # clean up the prediction file
        safe_remove(self.prediction_file)


    def get_model_file(self):
        return os.path.join(self.working_directory, self.filename)

    def get_cache_file(self):
        return os.path.join(self.working_directory, '%s.cache' % (self.handle))

    def get_prediction_file(self):
        return os.path.join(self.working_directory, '%s.prediction' % (self.handle))

    def get_example_file(self, suffix = ""):
        return os.path.join(self.working_directory, '%s.examples.%s' % (self.handle, suffix))
