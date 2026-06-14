import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock
import os
import time
import cv2
import gc
import base64
import asyncio
from PIL import Image
import datetime
import pickle as pkl
import random
import cv2
import warnings
from RRM.preference_label import PreAgent
from RRM.conv_net import CNN, fanin_init

device = 'cuda'


def gen_image_net(image_height, image_width, 
                  conv_kernel_sizes=[5, 3, 3 ,3], 
                  conv_n_channels=[16, 32, 64, 128], 
                  conv_strides=[3, 2, 2, 2]):
    conv_args=dict( # conv layers
        kernel_sizes=conv_kernel_sizes, # for sweep into, cartpole, drawer open. 
        n_channels=conv_n_channels,
        strides=conv_strides,
        output_size=1,
    )
    conv_kwargs=dict(
        hidden_sizes=[], # linear layers after conv
        batch_norm_conv=False,
        batch_norm_fc=False,
    )

    return CNN(
        **conv_args,
        paddings=np.zeros(len(conv_args['kernel_sizes']), dtype=np.int64),
        input_height=image_height,
        input_width=image_width,
        input_channels=3,
        init_w=1e-3,
        hidden_init=fanin_init,
        **conv_kwargs
    )

 # for softgym task
def gen_image_net2(output_range=None):

    class MYResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1)
            self.activation = nn.Tanh()
            print("Using activation: Tanh")
        def forward(self, x):
            out = self.backbone(x)
            out = self.activation(out)
            return out
    return MYResNet()


class RewardModel:
    def __init__(self, ds, da, 
                 ensemble_size=3, lr=3e-4, mb_size = 128, size_segment=1, 
                 max_size=100, activation='tanh', capacity=5e5,  
                 large_batch=1, label_margin=0.0, 
                 teacher_beta=-1, teacher_gamma=1, 
                 teacher_eps_mistake=0, 
                 teacher_eps_skip=0, 
                 teacher_eps_equal=0,
                 
                # vlm related params
                env_name="CartPole-v1",
                log_dir=None,

                # image based reward
                image_height=128,
                image_width=128,
                resize_factor=1,
                resnet=False,
                conv_kernel_sizes=[5, 3, 3 ,3],
                conv_n_channels=[16, 32, 64, 128],
                conv_strides=[3, 2, 2, 2],
                cached_label_path=None,
                kl_weight=5, 
                **kwargs
                ):
        
        # train data is trajectories, must process to sa and s..   
        self.ds = ds
        self.da = da
        # labels data format
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.size_segment = size_segment
        self.max_size = max_size
        self.capacity = int(capacity)

        self.buffer_seg1 = np.empty((self.capacity, size_segment, image_height, image_width, 3), dtype=np.uint8)
        self.buffer_seg2 = np.empty((self.capacity, size_segment, image_height, image_width, 3), dtype=np.uint8)
        self.image_height = image_height
        self.image_width = image_width
        self.resize_factor = resize_factor

        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        if self.size_segment > 1:
            self.buffer_score_1 = np.empty((self.capacity, size_segment), dtype=np.float32)
            self.buffer_score_2 = np.empty((self.capacity, size_segment), dtype=np.float32)
        self.buffer_frm1 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        self.buffer_frm2 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)

        self.buffer_index = 0
        self.buffer_full = False

        # reward model
        self.de = ensemble_size
        self.lr = lr
        self.activation = activation
        # not image-based model
        self.reward_model_layers = 3
        self.reward_model_H = 256
        # image-based model
        self.resnet = resnet
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_n_channels = conv_n_channels
        self.conv_strides = conv_strides

        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.construct_ensemble()


        if not self.resnet:
            self.train_batch_size = 32
        else:
            self.train_batch_size = 8

        self.CEloss = nn.CrossEntropyLoss(reduction='mean')
        
        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0
        
        self.label_margin = label_margin
        self.label_target = 1 - 2*self.label_margin
        self.large_batch = large_batch

        # vlm label
        self.env_name = env_name
        self.vlm_label_acc = 0

        
        # load cached labels
        file_path = os.path.abspath(__file__)
        self.dir_path = os.path.dirname(file_path)
        self.cached_label_path = None if cached_label_path is None else "{}".format(cached_label_path)
        self.read_cache_idx = 0
        if self.cached_label_path is not None:
            print(f"Cached label path: {self.cached_label_path}")
            all_cached_labels = sorted(os.listdir(self.cached_label_path))
            self.all_cached_labels = [os.path.join(self.cached_label_path, x) for x in all_cached_labels]
        else: self.all_cached_labels = []
        
        if "metaworld" in self.env_name:
            episode_length = 500
        elif "softgym_RopeFlattenEasy" in self.env_name:
            episode_length = 40
        elif "softgym_PassWater" in self.env_name:
            episode_length = 75
        elif "softgym_ClothFoldDiagonal" in self.env_name:
            episode_length = 3

        self.inputs = np.empty((max_size, episode_length, self.ds + self.da), dtype=np.float32) # max_episode, episode_max_step, state
        self.targets = np.empty((max_size, episode_length, 1), dtype=np.float32)

        if "Rope" in self.env_name or "Water" in self.env_name:
            self.img_inputs = np.empty((max_size, episode_length, 720, 720, 3), dtype=np.uint8)
        else:
            self.img_inputs = np.empty((max_size, episode_length, image_height, image_width, 3), dtype=np.uint8)
            
        self.buffer_pre_idx = 0
        self.buffer_pre_full = False
        self.buffer_pre_step_idx = 0
        self.episode_length = episode_length

        # save label path
        self.logdir = log_dir
        self.label_save_path = os.path.join(self.logdir, f"vlm_label_set")
        if not os.path.exists(self.label_save_path):
            os.makedirs(self.label_save_path)
        else:
            self.client = PreAgent(self.env_name, size_segment, project_dir=self.dir_path)
       
        self.kl_weight = kl_weight 

        self.preference_data_emptyed = False
        
    def eval(self,):
        for i in range(self.de):
            self.ensemble[i].eval()

    def train(self,):
        for i in range(self.de):
            self.ensemble[i].train()
    
    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax (input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]
    
    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size*new_frac)
    
    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)
        
    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip  # percent of an average segment reward
        
    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal  # percent of an average segment reward
        
    def construct_ensemble(self):
        for _ in range(self.de):
            
            if not self.resnet:
                model = gen_image_net(self.image_height, self.image_width, self.conv_kernel_sizes, self.conv_n_channels, self.conv_strides).float().to(device)
            else:
                model = gen_image_net2().float().to(device)
                
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
            
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
            
    def add_data(self, obs, act, rew, done, img=None):
        sa_t = np.concatenate([obs, act], axis=-1)
        r_t = np.array(rew)
        if img is not None:
            flat_img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

        np.copyto(self.inputs[self.buffer_pre_idx, self.buffer_pre_step_idx], sa_t)
        np.copyto(self.targets[self.buffer_pre_idx, self.buffer_pre_step_idx], r_t)
        if img is not None:
            np.copyto(self.img_inputs[self.buffer_pre_idx, self.buffer_pre_step_idx], flat_img)
        self.buffer_pre_step_idx += 1
        if done:
            self.buffer_pre_step_idx = 0
            self.buffer_pre_idx = (self.buffer_pre_idx + 1) % self.max_size
            self.buffer_pre_full = self.buffer_pre_full or self.buffer_pre_idx == 0
        
    def r_hat_member(self, x, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        return self.ensemble[member](torch.from_numpy(x).float().to(device))

    def r_hat(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    def r_hat_batch(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)
    
    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )
            
    def load(self, model_dir, step):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.join(file_dir, model_dir)
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
            )
    
    def get_queries(self, mb_size=20):
        max_len = self.max_size if self.buffer_pre_full else self.buffer_pre_idx
        # get train traj
        train_inputs = np.asarray(self.inputs[:max_len])
        train_targets = np.asarray(self.targets[:max_len])
        train_images = np.asarray(self.img_inputs[:max_len])

        
        # choose random episode
        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)

        # Generate time index
        sigment_time = np.arange(self.size_segment).reshape(1, -1)

        if 'Cloth' not in self.env_name:
            random_idx_2 = np.random.choice(self.episode_length-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
            time_index_2 = sigment_time + random_idx_2
            random_idx_1 = np.random.choice(self.episode_length-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
            time_index_1 = sigment_time + random_idx_1
        else:
            time_index_2 = sigment_time
            time_index_1 = sigment_time

        sa_t_1 = train_inputs[batch_index_1[:, None], time_index_1]  # mb_size, size_segment, dim of s&a
        r_t_1 = train_targets[batch_index_1[:, None], time_index_1]
        sa_t_2 = train_inputs[batch_index_2[:, None], time_index_2]  # mb_size, size_segment, dim of s&a
        r_t_2 = train_targets[batch_index_2[:, None], time_index_2]
 
        img_t_1 = train_images[batch_index_1[:, None], time_index_1]
        img_t_2 = train_images[batch_index_2[:, None], time_index_2]  # bath x segment x image_height x image_width x3
        
        return sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2

    def put_queries(self, sa_t_1, sa_t_2, labels, frm_1, frm_2, vlm_weights): # frm_1 data for reward check
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample

        # NOTE: When not using image based rewards, it gives concatenated state action pairs. When image based rewards are used, it gives the images.
        if next_index >= self.capacity:
            print("WARNING! label buffer full!")
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            
            sa_t_1 = sa_t_1.reshape(sa_t_1.shape[0], self.size_segment, sa_t_1.shape[2], sa_t_1.shape[3], sa_t_1.shape[4])
            sa_t_2 = sa_t_2.reshape(sa_t_2.shape[0], self.size_segment, sa_t_2.shape[2], sa_t_2.shape[3], sa_t_2.shape[4])

            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

            np.copyto(self.buffer_frm1[self.buffer_index:self.capacity], frm_1[:maximum_index])
            np.copyto(self.buffer_frm2[self.buffer_index:self.capacity], frm_2[:maximum_index])
            if self.size_segment > 1:
                np.copyto(self.buffer_score_1[self.buffer_index:self.capacity], vlm_weights[:maximum_index, 0, :])
                np.copyto(self.buffer_score_2[self.buffer_index:self.capacity], vlm_weights[:maximum_index, 1, :])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

                np.copyto(self.buffer_frm1[:remain], frm_1[maximum_index:])
                np.copyto(self.buffer_frm2[:remain], frm_2[maximum_index:])
                if self.size_segment > 1:
                    np.copyto(self.buffer_score_1[:remain], vlm_weights[maximum_index:, 0, :])
                    np.copyto(self.buffer_score_2[:remain], vlm_weights[maximum_index:, 1, :])

            self.buffer_index = remain
        else:
            if self.size_segment > 1:
                sa_t_1 = sa_t_1.reshape(sa_t_1.shape[0], self.size_segment, sa_t_1.shape[2], sa_t_1.shape[3], sa_t_1.shape[4])
                sa_t_2 = sa_t_2.reshape(sa_t_2.shape[0], self.size_segment, sa_t_2.shape[2], sa_t_2.shape[3], sa_t_2.shape[4])
            else:
                sa_t_1 = sa_t_1.reshape(sa_t_1.shape[0], 1, sa_t_1.shape[1], sa_t_1.shape[2], sa_t_1.shape[3])
                sa_t_2 = sa_t_2.reshape(sa_t_2.shape[0], 1, sa_t_2.shape[1], sa_t_2.shape[2], sa_t_2.shape[3])
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            
            np.copyto(self.buffer_frm1[self.buffer_index:next_index], frm_1)
            np.copyto(self.buffer_frm2[self.buffer_index:next_index], frm_2)
            if self.size_segment > 1:
                np.copyto(self.buffer_score_1[self.buffer_index:next_index], vlm_weights[:, 0, :])
                np.copyto(self.buffer_score_2[self.buffer_index:next_index], vlm_weights[:, 1, :])
            
            self.buffer_index = next_index
            
    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1=None, img_t_2=None):
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        # gt label
        # time decrease
        if self.teacher_gamma != 1:
            seg_size = r_t_1.shape[1]
            temp_r_t_1 = r_t_1.copy()
            temp_r_t_2 = r_t_2.copy()
            for index in range(seg_size-1):
                temp_r_t_1[:,:index+1] *= self.teacher_gamma
                temp_r_t_2[:,:index+1] *= self.teacher_gamma
            sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
            sum_r_t_2 = np.sum(temp_r_t_2, axis=1)
        
        # skip or equal data will not be quried with vlm
        if self.teacher_thres_skip > 0 or self.teacher_thres_equal > 0: 
            valid_indices = np.ones_like(r_t_1).reshape(-1)
            
            # skip the query
            if self.teacher_thres_skip > 0:
                max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
                # if > thres, set label index = 1, valid
                max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
                valid_indices = np.logical_and(valid_indices, max_index)

            # equally preferable
            if self.teacher_thres_equal > 0:
                # if < thres, set label index = 1, invalid
                margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)
                valid_indices = np.logical_and(valid_indices, not margin_index)
                # labels[margin_index] = -1

            if sum(valid_indices) == 0:
                warnings.warn("no valid labels!")
                return None, None, None, None, []

            sa_t_1 = sa_t_1[valid_indices.astype(bool)]
            sa_t_2 = sa_t_2[valid_indices.astype(bool)]
            r_t_1 = r_t_1[valid_indices.astype(bool)]
            r_t_2 = r_t_2[valid_indices.astype(bool)]
            
            img_t_1 = img_t_1[valid_indices.astype(bool)]
            img_t_2 = img_t_2[valid_indices.astype(bool)]

            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        rational_labels = 1*(sum_r_t_1 < sum_r_t_2)
        
        if self.teacher_beta > 0: # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1), torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat*self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels
        
        if self.teacher_eps_mistake > 0:
            # making a mistake
            len_labels = labels.shape[0]
            rand_num = np.random.rand(len_labels)
            noise_index = rand_num <= self.teacher_eps_mistake
            labels[noise_index] = 1 - labels[noise_index]
        
        # vlm label
        start_time = time.time()
        time_string = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d-%H-%M-%S')

        print(f"Using online vlm labels for this query!")
        vlm_labels = []
        vlm_weights = []
        for idx, (video1, video2) in enumerate(zip(img_t_1, img_t_2)):
            print("Querying gemini: {}/{} cost time:{:.2f}".format(idx, len(labels), time.time() - start_time))
            diff = 0
            for idx_diff in range(len(video1)):
                diff += np.linalg.norm(video1[idx_diff] - video2[idx_diff])
            if diff < 1e-3 * len(video1):
                print("Too similar, skip query!")
                vlm_labels.append(-1)
                vlm_weights.append([None, None])
                continue
            else:
                label, vlm_weights_0, vlm_weights_1 = self.client.video_preference_label(video1, video2, time_string, idx)
                vlm_labels.append(label)
                vlm_weights.append([vlm_weights_0, vlm_weights_1])
        
        # remove the vlm label=-1
        sa_t_1, sa_t_2, r_t_1, r_t_2, rational_labels, vlm_labels, vlm_weights, img_t_1, img_t_2 = self.filter_useless_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, rational_labels, vlm_labels, vlm_weights, img_t_1, img_t_2
        )

        combined_images_list = np.concatenate([img_t_1, img_t_2], axis=3)  # along image_width axis for save labels
        with open("{}/{}.pkl".format(self.label_save_path, time_string), "wb") as f:
            pkl.dump([combined_images_list, rational_labels, vlm_labels, vlm_weights, sa_t_1, sa_t_2, r_t_1, r_t_2], f, protocol=pkl.HIGHEST_PROTOCOL)
            
        query_label_num = len(labels)
        acc = 0
        useful_label_num = len(rational_labels)
        if useful_label_num > 0:
            acc = np.sum(vlm_labels == rational_labels) / useful_label_num
            print("useful/query: {}/{} vlm label acc: {:.3f} ".format(useful_label_num, query_label_num, acc))
            print("useful/query: {}/{} vlm label acc: {:.3f} ".format(useful_label_num, query_label_num, acc))
        else:
            print("no vlm label")
            print("no vlm label")

        self.vlm_label_acc = acc

        return sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, rational_labels, vlm_labels, vlm_weights


    def filter_useless_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2, rational_labels, vlm_labels, vlm_weights, img_t_1, img_t_2):
        vlm_labels = np.array(vlm_labels)
        good_idx = vlm_labels != -1
        
        sa_t_1 = sa_t_1[good_idx]
        sa_t_2 = sa_t_2[good_idx]
        r_t_1 = r_t_1[good_idx]
        r_t_2 = r_t_2[good_idx]
        rational_labels = rational_labels[good_idx]
        vlm_labels = vlm_labels[good_idx].reshape(-1, 1)
        vlm_weights = [row for i, row in enumerate(vlm_weights) if good_idx[i]]
        vlm_weights = np.array(vlm_weights)
        img_t_1 = img_t_1[good_idx]
        img_t_2 = img_t_2[good_idx]

        return sa_t_1, sa_t_2, r_t_1, r_t_2, rational_labels, vlm_labels, vlm_weights, img_t_1, img_t_2
    
    def uniform_sampling(self):

        if self.cached_label_path is None or self.read_cache_idx >= len(self.all_cached_labels):
            # use online vlm labels
            sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2 =  self.get_queries(mb_size=self.mb_size)

            sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, gt_labels, vlm_labels, vlm_weights = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2)
        
        else:
            # use cached vlm labels
            if self.read_cache_idx < len(self.all_cached_labels):
                combined_images_list, sa_t_1, sa_t_2, r_t_1, r_t_2, gt_labels, vlm_labels, vlm_weights = self.get_label_from_cached_states()
                if self.size_segment == 1:
                    num, height, width, _ = combined_images_list.shape
                    img_t_1 = combined_images_list[:, :, :width//2, :]
                    img_t_2 = combined_images_list[:, :, width//2:, :]
                else:
                    num, segment, height, width, _ = combined_images_list.shape
                    img_t_1 = combined_images_list[:, :, :, :width//2, :]
                    img_t_2 = combined_images_list[:, :, :, width//2:, :]
            else:
                vlm_labels = []
            
        labels = vlm_labels
            
        if len(labels) > 0:

            if self.size_segment == 1:
                self.put_queries(img_t_1[:, ::self.resize_factor, ::self.resize_factor, :], img_t_2[:, ::self.resize_factor, ::self.resize_factor, :], labels, sa_t_1, sa_t_2, vlm_weights)

            else:
                self.put_queries(img_t_1[:, :, ::self.resize_factor, ::self.resize_factor, :], img_t_2[:, :, ::self.resize_factor, ::self.resize_factor, :], labels, sa_t_1, sa_t_2, vlm_weights)

        return len(labels)
    
    def get_label_from_cached_states(self):
        if self.read_cache_idx >= len(self.all_cached_labels):
            return None, None, None, None, None, []
            
        with open(self.all_cached_labels[self.read_cache_idx], 'rb') as f:
            data = pkl.load(f)
        if self.size_segment == 1:
            # for single image
            combined_images_list, rational_labels, vlm_labels, sa_t_1, sa_t_2, r_t_1, r_t_2 = data
            vlm_weights = None
        else:
            combined_images_list, rational_labels, vlm_labels, vlm_weights, sa_t_1, sa_t_2, r_t_1, r_t_2 = data
        self.vlm_label_acc = np.sum(vlm_labels == rational_labels) / len(vlm_labels)
        print(f"Using cached labels for this query! vlm label num: {len(vlm_labels)} acc:{self.vlm_label_acc:.3f}\n {self.all_cached_labels[self.read_cache_idx]}")
        self.read_cache_idx += 1
        return combined_images_list, sa_t_1, sa_t_2, r_t_1, r_t_2, rational_labels, vlm_labels, vlm_weights
    
    def train_reward(self):
        ensemble_losses = np.array([0.0 for _ in range(self.de)])
        ensemble_kl_losses = np.array([0.0 for _ in range(self.de)])
        ensemble_ce_losses = np.array([0.0 for _ in range(self.de)])
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            # shuffle label data
            total_batch_index.append(np.random.permutation(max_len))
        
        num_iters = int(np.ceil(max_len/self.train_batch_size))
        total = 0

        for iter_id in range(num_iters):

            self.opt.zero_grad()
            total_loss = 0.0
            
            last_index = (iter_id+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][iter_id*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                if self.kl_weight != 0:
                    score_1 = self.buffer_score_1[idxs]
                    score_2 = self.buffer_score_2[idxs]
                    
                    score_1 = torch.from_numpy(score_1).float().to(device)
                    score_2 = torch.from_numpy(score_2).float().to(device)
                
                if member == 0:
                    total += labels.size(0)
                
                batch, time_step, H, W, C = sa_t_1.shape
                # sa_t_1 is batch_size x segment x image_height x image_width x 3
                sa_t_1 = np.transpose(sa_t_1, (0, 1, 4, 2, 3)).reshape(-1, C, H, W) # for torch we need to transpose channel first
                sa_t_2 = np.transpose(sa_t_2, (0, 1, 4, 2, 3)).reshape(-1, C, H, W)
                # also we stored uint8 images, we need to convert them to float32
                sa_t_1 = sa_t_1.astype(np.float32) / 255.0
                sa_t_2 = sa_t_2.astype(np.float32) / 255.0

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member).view(batch, time_step, 1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member).view(batch, time_step, 1)

                # compute CE loss
                r_hat1_ce = r_hat1.sum(axis=1)
                r_hat2_ce = r_hat2.sum(axis=1)
                r_hat_ce = torch.cat([r_hat1_ce, r_hat2_ce], axis=-1)
                CE_loss = self.CEloss(r_hat_ce, labels)
                ensemble_ce_losses[member] += CE_loss.item()

                # compute KL loss
                q_hat1 = F.log_softmax(r_hat1.squeeze(-1), dim=1)
                q_hat2 = F.log_softmax(r_hat2.squeeze(-1), dim=1)
                p_1 = F.softmax(score_1, dim=1).detach()
                p_2 = F.softmax(score_2, dim=1).detach()
                KL_loss = (F.kl_div(q_hat1, p_1, reduction='batchmean') + F.kl_div(q_hat2, p_2, reduction='batchmean')) * self.kl_weight
                # compute loss
                member_loss = CE_loss + KL_loss  # batch mean loss
                ensemble_kl_losses[member] += KL_loss.item()
 
                total_loss += member_loss

                # compute acc and loss
                _, predicted = torch.max(r_hat_ce.data, 1)
                correct = (predicted == labels).sum().item()

                ensemble_acc[member] += correct
                ensemble_losses[member] += member_loss.item()
            total_loss.backward()
            self.opt.step()
        # acc is the sum of all epoch samples
        ensemble_acc = ensemble_acc / total
        # loss is the average of batch loss in every iteration. the ensemble loss is the sum of all iteration loss
        ensemble_losses = ensemble_losses / num_iters
        ensemble_ce_losses = ensemble_ce_losses / num_iters
        ensemble_kl_losses = ensemble_kl_losses / num_iters

        torch.cuda.empty_cache()
        print(f"ACC: {ensemble_acc.mean():.3f}\tAVERAGE LOSS: {ensemble_losses.mean():.4f}\tCE: {ensemble_ce_losses.mean():.4f}\tKL:{ensemble_kl_losses.mean():.4f}")
        return ensemble_acc
    
    def empty_preference_data(self):

        del self.buffer_seg1, self.buffer_seg2, self.buffer_label, self.buffer_score_1, self.buffer_score_2, self.buffer_frm1, self.buffer_frm2
        del self.inputs, self.targets
        del self.img_inputs
        gc.collect()
        print("Preference data set EMPTYED!!")
        self.preference_data_emptyed = True
