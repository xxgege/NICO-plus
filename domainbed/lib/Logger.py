import logging
import os
import time

# set logger
class Logger():
    def __init__(self, args, hparams):
        self.txtname = time.strftime('%Y_%m%d_%H%M%S', time.localtime(time.time()))
        self.expname = self.get_expname(args, hparams)
        self.logpath = os.path.join(args.log_dir, self.expname, self.txtname)
        os.makedirs(os.path.join(args.log_dir, self.expname), exist_ok=True)

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO) # Here was logging.DEBUG...
        self.fh = logging.FileHandler(self.logpath, mode='a')
        self.fh.setLevel(logging.INFO)
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)
        self.logger.addHandler(self.fh)
        self.logger.addHandler(self.ch)

    def log_args(self, args):
        for k, v in vars(args).items():
            self.logger.info('%s: %s' % (k, str(v)))
        self.logger.info("")
    
    def debug(self, string):
        self.logger.debug(string)

    def info(self, string):
        self.logger.info(string)

    @staticmethod
    def get_expname(args, hparams):
        pretrain = "pretrain" if hparams["pretrain"] else  "nopretrain"
        return  "%s_%s_%s_to_%s_%s" % (args.dataset, args.algorithm, "-".join(args.source), "-".join(args.target), pretrain)
        