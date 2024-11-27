from .base import Base
import sys
# sys.path.append("/home/xxxx/BackdoorBox/core/defenses")
# sys.path.append("/home/xxxx/BackdoorBox/core/defenses/LAVA")
import LAVA.lava as lava
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader

class LAVA(Base):
    def __init__(self, args,
                 poisoned_testset, clean_testset, clean_validation, device, seed=0,
                 deterministic=False):
        super(LAVA, self).__init__(seed, deterministic)
        self.poisoned_testset = poisoned_testset
        self.clean_testset = clean_testset
        self.clean_validation = clean_validation
        self.device = device
        self.args = args

    def verify(self):
        training_size = len(self.poisoned_testset) * 2
        resize = 64 if self.args == "ImageNet_Subset" else 32

        loaders = {}
        test_dataset = ConcatDataset([self.poisoned_testset, self.clean_testset])
        test_dataset.targets = self.poisoned_testset.targets + self.clean_testset.targets

        print(len(test_dataset.targets))


        loaders["test"] = DataLoader(test_dataset, batch_size=64, shuffle=False)
        loaders["val"] = DataLoader(self.clean_validation, batch_size=64, shuffle=False)

        # for img, target in self.clean_testset:
        #     print(target)
        #     break

        feature_extractor = lava.load_pretrained_feature_extractor('cifar10_embedder_preact_resnet18.pth', self.device)

        dual_sol = lava.compute_dual(feature_extractor, loaders['test'], loaders['val'],
                                                        training_size, resize=resize)

        calibrated_gradient_poi = lava.compute_values_and_visualize(dual_sol, training_size)

        print(len(calibrated_gradient_poi))

        # loaders["test"] = DataLoader(self.clean_testset, batch_size=64, shuffle=False)
        #
        # dual_sol = lava.compute_dual(feature_extractor, loaders['test'], loaders['val'],
        #                              training_size, resize=resize)
        #
        # calibrated_gradient_clean = lava.compute_values_and_visualize(dual_sol, training_size)
        #
        # print(len(calibrated_gradient_clean))

        return list(calibrated_gradient_poi[:len(self.poisoned_testset)]), list(calibrated_gradient_poi[len(self.poisoned_testset):])


