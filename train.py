#! python3
# -*- coding: utf-8 -*-
"""
################################################################################################
Implementation of 'PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION'##
https://arxiv.org/pdf/1710.10196.pdf                                                          ##
################################################################################################
https://github.com/shanexn
Created: 2018-06-18
################################################################################################
"""
import os
import random
import warnings
from PIL import Image
from model import *
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

import torch.utils.data as udata
import torchvision.datasets as vdatasets
import torchvision.transforms as transforms

################################################################################
MANUAL_SEED = 0.5
CUDNN_BENCHMARK = True
TO_GPU = True  # For now, all run in GPU mode
DEVICE = torch.device("cuda:0")
G_LR = 0.0001
D_LR = 0.0001
ADAM_BETA = (0.0, 0.99)
ADAM_EPS = 1e-8
LOD_STABLE_IMGS = 2 * 1e5   # Thousands of real images to show before doubling the resolution.
LOD_FADEIN_IMGS = 2 * 1e5   # Thousands of real images to show when fading in new layers.
INIT_RESOLUTION = 4         # Image resolution used at the beginning training.
SAVE_IMG_STEP = 500
LAMBDA_FOR_WGANGP = 10
CRITIC_FOR_WGANGP = 1
MINIBATCH_MAPPING = {
    4: 16,
    8: 16,
    16: 16,
    32: 16,
    64: 16,
    128: 8,
    256: 4,
    512: 2,
    1024: 1
}


#################################################################################
# Construct Generator and Discriminator #########################################
#################################################################################
class PGGAN(object):
    def __init__(self,
                 resolution,            # Resolution.
                 latent_size,           # Dimensionality of the latent vectors.
                 criterion_type="GAN",  # ["GAN", "WGAN-GP"]
                 minibatch_mapping=MINIBATCH_MAPPING,  # Batch size of each resolution
                 rgb_channel=3,         # Output channel size, for rgb always 3
                 fmap_base=2 ** 13,     # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,        # log2 feature map reduction when doubling the resolution.
                 fmap_max=2 ** 9,       # Maximum number of feature maps in any layer.
                 is_tanh=True,
                 is_sigmoid=True
                 ):
        self.latent_size_ = latent_size
        self.rgb_channel_ = rgb_channel
        self.fmap_base_ = fmap_base
        self.fmap_decay_ = fmap_decay
        self.fmap_max_ = fmap_max

        self.image_pyramid_ = int(np.log2(resolution))  # max level of the Image Pyramid
        self.resolution_ = 2 ** self.image_pyramid_     # correct resolution
        self.lod_init_res_ = INIT_RESOLUTION  # Image resolution used at the beginning.

        self.criterion_type_ = criterion_type
        self.is_tanh_ = is_tanh
        self.is_sigmoid_ = False if self.criterion_type_ in ["WGAN-GP"] else is_sigmoid

        self.gradient_weight_real_ = torch.FloatTensor([-1]).cuda()
        self.gradient_weight_fake_ = torch.FloatTensor([1]).cuda()

        # Mentioned on page13 of the Paper (Appendix A.1)
        # 4x4 - 128x128 -> 16
        # 256x256       -> 14
        # 512x512       -> 6
        # 1024x1024     -> 3
        # self.minibatch_mapping_ = {
        #     4: 16, 8: 16, 16: 16, 32: 16, 64: 16, 128: 16,
        #     256: 16, 512: 6, 1024: 3}
        self.minibatch_mapping_ = minibatch_mapping

        self._init_network()

    def gen_image_by_existed_models(self, model_path, file_name, fake_seed=None):

        import re
        regx = "(?<=x)\d+"

        if isinstance(model_path, (tuple, list)):
            g_model_path, d_model_path = model_path
        else:
            g_model_path = model_path
        if not os.path.exists(g_model_path):
            raise FileExistsError("Models not exist: {}".format(model_path))

        res = re.findall(regx, g_model_path)
        if res is None:
            raise TypeError("Models is not inappropriate: {}".format(model_path))
        res = int(res[0])
        # batch_size = self.minibatch_mapping_[res]
        batch_size = 1
        self.g_net.net_config = [int(np.log2(res))-2, "stable", 1.0]
        self.g_net.load_state_dict(torch.load(g_model_path))
        # self.d_net.load_state_dict(torch.load(d_model_path))
        if fake_seed is None:
            fake_seed = torch.randn(batch_size,
                                    self.latent_size_, 1, 1)
        fake_work = self.g_net(fake_seed)
        vutils.save_image(fake_work.detach(),
                          file_name, nrow=4, normalize=True, padding=0)

    def train(self, dataset_root):
        # init settings
        random.seed = MANUAL_SEED
        torch.manual_seed(MANUAL_SEED)
        cudnn.benchmark = CUDNN_BENCHMARK
        # self._init_network()
        self._init_optimizer()
        self._init_criterion()

        net_level = 0
        net_status = "stable"
        net_alpha = 1.0
        pyramid_from = int(np.log2(self.lod_init_res_))
        pyramid_to = self.image_pyramid_
        for cursor_pyramid in range(pyramid_from-1, pyramid_to):
            minibatch = self.minibatch_mapping_[2**(cursor_pyramid+1)]  # if not in the map, raise error directly
            net_level = cursor_pyramid - 1
            stable_steps = int(LOD_STABLE_IMGS // minibatch)
            fadein_steps = int(LOD_FADEIN_IMGS // minibatch)

            current_level_res = 2 ** (net_level + 2)

            d_datasets = collect_image_data(dataset_root, minibatch, current_level_res, 2)
            fake_seed = torch.randn(minibatch, self.latent_size_, 1, 1).cuda()

            # The first step should be keep "stable" status
            # The rest steps would run as two phases, "fadein" -> "stable"
            if cursor_pyramid == pyramid_from-1:
                net_status == "stable"
                for cursor_step in range(stable_steps):
                    self._train(net_level, net_status, net_alpha, minibatch, cursor_step, d_datasets, fake_seed)
            else:
                net_status = "fadein"
                for cursor_step in range(fadein_steps):
                    net_alpha = 1.0 - (cursor_step + 1) / fadein_steps
                    self._train(net_level, net_status, net_alpha, minibatch, cursor_step, d_datasets, fake_seed)
                for cursor_step in range(stable_steps):
                    net_alpha = 1.0
                    self._train(net_level, "stable", net_alpha, minibatch, cursor_step, d_datasets, fake_seed)
            torch.save(self.g_net.state_dict(), './output/Gnet_%dx%d.pth' % (current_level_res, current_level_res))
            torch.save(self.d_net.state_dict(), './output/Dnet_%dx%d.pth' % (current_level_res, current_level_res))

    def _train(self, net_level, net_status, net_alpha, batch_size, cur_step, d_datasets, fake_seed):
        # prepare data
        current_level_res = 2 ** (net_level + 2)
        g_data = fake_seed
        work_real = random.choice(d_datasets)[0].cuda()

        # init nets
        self.g_net.net_config = [net_level, net_status, net_alpha]
        self.d_net.net_config = [net_level, net_status, net_alpha]

        work_fake = self.g_net(g_data)
        d_real = self.d_net(work_real)
        d_fake = self.d_net(work_fake.detach())

        if self.criterion_type_ == "WGAN-GP":
            # update d_net
            self.d_net.zero_grad()
            # # for i in range(CRITIC_FOR_WGANGP):
            mean_real = d_real.mean()
            mean_real.backward(self.gradient_weight_real_)
            mean_fake = d_fake.mean()
            mean_fake.backward(self.gradient_weight_fake_)

            gradient_penalty = self._gradient_penalty(work_real.data, work_fake.data, batch_size, current_level_res)
            gradient_penalty.backward()
            self.d_optim.step()

            # update g_net
            self.g_net.zero_grad()
            d_fake = self.d_net(work_fake)
            mean_fake = d_fake.mean()
            mean_fake.backward(self.gradient_weight_real_)
            self.g_optim.step()

        elif self.criterion_type_ == "GAN":
            target_real = torch.full((batch_size, 1, 1, 1), 1).cuda()
            target_fake = torch.full((batch_size, 1, 1, 1), 0).cuda()
            # update d_net
            self.d_net.zero_grad()
            loss_d_real = self.criterion(d_real, target_real)
            loss_d_real.backward()
            loss_d_fake = self.criterion(d_fake, target_fake)
            loss_d_fake.backward()
            # loss_d = loss_d_real + loss_d_fake * 0.1
            # loss_d.backward()
            self.d_optim.step()

            # update g_net
            self.g_net.zero_grad()
            d_fake = self.d_net(work_fake)
            loss_g = self.criterion(d_fake, target_real)
            loss_g.backward()
            self.g_optim.step()

        if cur_step % SAVE_IMG_STEP == 0:
            vutils.save_image(work_real,
                              "./output/real_netlevel%02d.jpg" % net_level,
                              nrow=4, normalize=True, padding=0)
            fake_work = self.g_net(g_data)
            vutils.save_image(fake_work.detach(),
                              './output/fake_%s_netlevel%02d_%07d.jpg' % (net_status, net_level, cur_step),
                              nrow=4, normalize=True, padding=0)

    def _init_network(self):
        # Init Generator and Discriminator
        self.g_net = Generator(self.resolution_, self.latent_size_, self.rgb_channel_,
                               self.fmap_base_, self.fmap_decay_, self.fmap_max_, is_tanh=self.is_tanh_)
        self.d_net = Discriminator(self.resolution_, self.rgb_channel_,
                                   self.fmap_base_, self.fmap_decay_, self.fmap_max_, is_sigmoid=self.is_sigmoid_)
        # if TO_GPU:
        self.g_net.cuda()
        self.d_net.cuda()

    def _init_optimizer(self):
        self.g_optim = optim.Adam(self.g_net.parameters(), lr=G_LR, betas=ADAM_BETA, eps=ADAM_EPS)
        self.d_optim = optim.Adam(self.d_net.parameters(), lr=D_LR, betas=ADAM_BETA, eps=ADAM_EPS)

    def _init_criterion(self):
        if self.criterion_type_ == "GAN":
            self.criterion = nn.BCELoss()
        elif self.criterion_type_ == "WGAN-GP":
            self.criterion = self._gradient_penalty
        else:
            raise ValueError("INVALID TYPE: {0}.".format(self.criterion_type_))

    def _gradient_penalty(self, real_data, fake_data, batch_size, res):
        """
        This algorithm was mentioned on the Page4 of paper
        'Improved Training of Wasserstein GANs'
        This implementation was from 'https://github.com/caogang/wgan-gp'
        """
        epsilon = torch.rand(batch_size, 1)
        epsilon = epsilon.expand(batch_size, real_data.nelement() / batch_size).contiguous().view(batch_size, 3, res, res)
        epsilon = epsilon.cuda()
        median_x = epsilon * real_data + ((1 - epsilon) * fake_data)

        # if TO_GPU:
        median_x = median_x.cuda()
        median_data = torch.autograd.Variable(median_x, requires_grad=True)

        d_median_data = self.d_net(median_data)

        gradients = torch.autograd.grad(outputs=d_median_data, inputs=median_data,
                                        grad_outputs=torch.ones(d_median_data.size()).cuda(),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA_FOR_WGANGP
        return gradient_penalty


#################################################################################
# Image loading and fake data generating ########################################
#################################################################################
# Temp used, will be rewritten
def collect_image_data(dir, batch_size, resolution, num_workers, max_samplesize=150):
    path = os.path.join(dir, str(resolution))
    dset = vdatasets.ImageFolder(
        root=path, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = udata.DataLoader(dset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, drop_last=True)
    output = []
    for i, j in enumerate(dataloader):
        if i == max_samplesize:
            break
        output.append(j)

    return output


def collect_fake_data(tag_dir):
    pass


#################################################################################
# Func for reorganising images ##################################################
#################################################################################
def gen_classified_images(dir_path, classes=None, centre_crop=True, save_to_local=False):
    """
    dir_path
        |_ img1
        |_ img2
    :return:
    """
    classes = [4, 8, 16, 32, 64, 128, 256, 512, 1024] if classes is None else classes
    image_dataset = {i: [] for i in classes}
    for i in os.listdir(dir_path):
        img_path = os.path.join(dir_path, i)
        if not os.path.isfile(img_path):
            continue
        # im = _image_reader(os.path.join(dir_path, i))
        im = Image.open(os.path.join(dir_path, i))
        im_info = im.info
        if im is None:
            continue
        o_width, o_height = im.size
        if centre_crop is True:
            min_dim = o_width if o_width < o_height else o_height
            im = _image_centre_crop(im, (min_dim, min_dim))

        for class_ in classes:
            im_resized = _resize_image(im, (class_, class_))
            image_dataset[class_].append(im_resized)
            if save_to_local is True:
                folder = os.path.join(os.path.dirname(dir_path),
                                      "{0}_{1}".format(os.path.basename(dir_path), "classified"),
                                      str(class_), str(class_))
                os.makedirs(folder, exist_ok=True)
                im_resized.save(os.path.join(folder, i), **im_info)
    return image_dataset


def _resize_image(im, size, resample=Image.ANTIALIAS):
    return im.resize(size, resample)


def _image_centre_crop(im, outsize):
    im_width, im_height = im.size
    out_width, out_height = outsize
    left = int(round((im_width - out_width) / 2.))
    upper = int(round((im_height - out_height) / 2.))
    right = left + out_width
    lower = upper + out_height
    return im.crop((left, upper, right, lower))


if __name__ == "__main__":
    p = PGGAN(1024, 512, "WGAN-GP")
    p.train(r"E:\workspace\datasets\CelebA\Img\img_align_celeba_classified")
    # p.gen_image_by_existed_models(r"E:\workspace\gitrepo\gitlab\x-Gans\pggan\output\Gnet_128x128.pth",
    #                               r"E:\workspace\gitrepo\gitlab\x-Gans\pggan\test001.jpg")
    # a = collect_d_data(r"E:\workspace\datasets\CelebA\Img\img_align_celeba_classified", 1, 1024, 2)

