from LEEMcontrol import oLeem
from UVIEWcontrol import oUview
import check_resolution
import logging
from skimage import io
import the_watchdog as watch
import time
from tools import rawToImage
import numpy as np
import tensorflow as tf
from tools import stepChanges


class LEEM_remote(object):
    def __init__(self):
        self.LEEM = oLeem(port=5566, ip='localhost')
        self.LEEM.connect()
        self.LEEM.testConnect()
        self.modules = self._findModules()
        self.state = []
        self.configuration = self._collectConfiguration()
        self.n_actions = 2 * len(self.modules)
        self.change = stepChanges(self.modules)
        assert len(self.change) == len(self.modules)

        self.image_quality = check_resolution.ImageEvaluation()

        self.Uview = oUview(port=5570, ip='localhost')
        self.Uview.connect()

        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%I:%M:%S',
                            level=logging.INFO,
                            handlers=[logging.FileHandler("logging.log"), logging.StreamHandler()],
                            )
        self.template = " changed from {} mA to {} mA"
        self.step_counter = 0

        # for Watchdog, which is a back-up method if Uview connection fails
        self.image_created_flag = False
        self.watchdog = None
        self.path = None
        self.latest_image = None

    def _findModules(self):
        modules = dict()
        i = 1
        for mnemonic in self.LEEM.Mnemonic.values():
            if isinstance(self.LEEM.getValue(mnemonic), float):
                if mnemonic != 'KSTEMP':  # atm, cant change sample temp
                    modules[i] = mnemonic
                    i += 1
        # l = len(modules)
        # for i, module in enumerate(self.LEEM.Module.values()):
        ## this gives same valuse as LEEM.Mnemonics,
        ## difference is in name of module and mnemonics only gives controllable values
        #     modules[i+l] = module

        return modules

    def _collectConfiguration(self):
        configuration = np.zeros((len(self.modules)))
        for i, module in enumerate(self.modules.values()):
            configuration[i] = self.LEEM.getValue(module)

        return configuration

    def _positive_step_module(self, module):
        current_value = self.LEEM.getValue(module)
        new_value = self.change[module] + current_value
        self.LEEM.setValue(module, new_value)
        logging.info("The module " + module + self.template.format(current_value, new_value))

    def _negative_step_module(self, module):
        current_value = self.LEEM.getValue(module)
        new_value = current_value - self.change[module]
        self.LEEM.setValue(module, new_value)
        logging.info("The module " + module + self.template.format(current_value, new_value))

    def _get_image(self):
        return self.Uview.getImage()

    @staticmethod
    def _image_to_tensor(image):
        return tf.expand_dims(tf.convert_to_tensor(image), axis=-1)

    def _add_to_state(self, image):
        self.state.append(self._image_to_tensor(image))

    def reset(self, rand=False):
        logging.info('LEEM RESET ----------')
        for i, module in enumerate(self.modules.values()):
            if rand:
                new_val = np.random.normal(loc=self.configuration[i], scale=abs(0.01 * self.configuration[i]))
            else:
                new_val = self.configuration[i]
            self.LEEM.setValue(module, new_val)
            logging.info("The module " + module + self.template.format(self.configuration[i], new_val))
            # print(module + ' changed from {} to {}'.format(self.configuration[i],
            #                                                np.random.normal(loc=self.configuration[i],
            #                                                                 scale=abs(0.01*(self.configuration[i])))))
        self._add_to_state(self._get_image())

    def action(self, action_key):
        """actions are integers from 2 to 2*(number of actions)+1, where an even number 2k indicates positive change
            in module k, odd number 2k+1 indicates negative change in module k"""
        if action_key % 2 == 0:
            self._positive_step_module(self.modules[int(action_key / 2)])
        else:
            self._negative_step_module(self.modules[int((action_key - 1) / 2)])

    def step(self, action):
        self.action(action_key=action)
        output_image = self._get_image()
        self.image_quality.inputImage(output_image)
        reward = self.image_quality.scoreFocus() / 1e6
        self._add_to_state(output_image)
        self.step_counter += 1
        logging.info('Reward: {}'.format(reward))
        return output_image, reward, self._isDone(reward)

    def _isDone(self, reward):
        return self.step_counter > 10000 or reward < 1e-2

    def print_state(self, save=False):
        for module in self.modules.values():
            print(module)
        if save:
            f = open("modules.txt", "w+")
            f.write(str(self.modules))
            f.close()

    """Watchdog methods: """

    @staticmethod
    def getImage(path):
        return rawToImage(path)

    def WDImageFound(self, image_path):
        self.latest_image = image_path
        self.image_created_flag = True

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
