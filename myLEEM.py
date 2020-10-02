from LEEMcontrol import oLeem
from UVIEWcontrol import oUview
import check_resolution
import logging
from skimage import io
import the_watchdog as watch
import time
from tools import rawToImage


class LEEM_remote(object):
    def __init__(self):
        self.LEEM = oLeem(port=5566, ip='localhost')
        self.LEEM.connect()
        self.LEEM.testConnect()
        self.modules = dict()
        self._findModules()
        self.n_actions = 2*len(self.modules)
        self.change = 1 / 100

        self.Uview = oUview()
        self.Uview.connect()

        self.logging.basicConfig(filename="logfile.log", level=logging.INFO)
        self.template = " changed from {} mA to {} mA"
        self.step_counter = 0

        # for Watchdog, which is a back-up method if Uview connection fails
        self.image_created_flag = False
        self.watchdog = None
        self.path = None
        self.latest_image = None

    def _findModules(self):
        for i, module in enumerate(self.LEEM.Modules.values()):
            self.modules[i] = module
            # what about mnemonics? do we have any of those???

    def positive_step_module(self, module):
        current_value = self.LEEM.getValue(module)
        new_value = (1 + self.change) * current_value
        self.LEEM.setValue(module, new_value)
        self.logging.info("The module " + module + self.template.format(current_value, new_value))

    def negative_step_module(self, module):
        current_value = self.LEEM.getValue(module)
        new_value = (1 - self.change) * current_value
        self.LEEM.setValue(module, new_value)
        self.logging.info("The module " + module + self.template.format(current_value, new_value))

    @staticmethod
    def getImage(path):
        return rawToImage(path)

    def WDImageFound(self, image_path):
        self.latest_image = image_path
        self.image_created_flag = True

    def action(self, action_key):
        """actions are integers from 2 to 2*(number of actions)+1, where an even number 2k indicates positive change
            in module k, odd number 2k+1 indicates negative change in module k"""
        if action_key % 2 == 0:
            self.positive_step_module(self.module[int(action_key / 2)])
        else:
            self.negative_step_module(self.module[int((action_key - 1) / 2)])

    def step(self, action):
        self.action(action_key=action)

        reward = check_resolution.ImageEvaluation(self.Uview.getImage())
        self.step_counter += 1
        return self.getImage(self.latest_image), reward, self._isDone()

    def _isDone(self):
        return self.step_counter > 10000

    def print_state(self):
        for i in self.LEEM.Modules.values():
            print(i)
        for i in self.LEEM.Mnemonic.values():
            print(i)

    """Watchdog methods: """

    def WDwatchdogReactivate(self):
        self.WDwatchForImage()
        while not self.image_created_flag:
            time.sleep(1)

    def WDsetFluffy(self, path=None):
        if path is not None:
            self.path = path
        self.watchdog = watch.Fluffy(self.WDImageFound, self.path)
        self.watchdog.SniffAround()

    def WDwatchForImage(self):
        self.image_created_flag = False
