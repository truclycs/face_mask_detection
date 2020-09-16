import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)






# import os
# import sys
# import time

# import tensorflow as tf
# from absl import flags, logging, app
# from absl.flags import FLAGS

# from components import config
# from components.lr_scheduler import MultiStepWarmUpLR
# from components.prior_box import priors_box
# from components.utils import set_memory_growth
# from dataset.tf_dataset_preprocess import load_dataset
# from network.losses import MultiBoxLoss
# from network.network import SlimModel

# flags.DEFINE_string('gpu', '0', 'which gpu to use')


# def main(_):
#     global load_t1
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#     os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu  # CPU:'-1'
#     logger = tf.get_logger()
#     logger.disabled = True
#     logger.setLevel(logging.FATAL)
#     set_memory_growth()

#     weights_dir = 'checkpoints/'
#     if not os.path.exists(weights_dir):
#         os.mkdir(weights_dir)
#     # if os.path.exists('logs'):
#     #     shutil.rmtree('logs')

#     logging.info("Load configuration...")
#     cfg = config.cfg
#     label_classes = cfg['labels_list']
#     logging.info(f"Total image sample:{cfg['dataset_len']},Total classes number:"
#                  f"{len(label_classes)},classes list:{label_classes}")

#     logging.info("Compute priors boxes...")
#     priors, num_cell = priors_box(cfg)
#     logging.info(f"Prior boxes number:{len(priors)},default anchor box number per feature map cell:{num_cell}")

#     logging.info("Loading dataset...")
#     train_dataset = load_dataset(cfg, priors, shuffle=True, train=True)
#     # val_dataset = load_dataset(cfg, priors, shuffle=False, train=False)

#     logging.info("Create Model...")
#     try:
#         model = SlimModel(cfg=cfg, num_cell=num_cell, training=True)
#         model.summary()
#         tf.keras.utils.plot_model(model, to_file=os.path.join(os.getcwd(), 'model.png'),
#                                   show_shapes=True, show_layer_names=True)
#     except Exception as e:
#         logging.error(e)
#         logging.info("Create network failed.")
#         sys.exit()

#     if cfg['resume']:
#         # Training from latest weights
#         paths = [os.path.join(weights_dir, path)
#                  for path in os.listdir(weights_dir)]
#         latest = sorted(paths, key=os.path.getmtime)[-1]
#         model.load_weights(latest)
#         init_epoch = int(os.path.splitext(latest)[0][-3:])

#     else:
#         # Training from scratch
#         init_epoch = -1

#     steps_per_epoch = cfg['dataset_len'] // cfg['batch_size']
#     # val_steps_per_epoch = cfg['val_len'] // cfg['batch_size']

#     logging.info(f"steps_per_epoch:{steps_per_epoch}")

#     logging.info("Define optimizer and loss computation and so on...")

#     # learning_rate =tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,
#     #                                                              decay_steps=20000,
#     #                                                              decay_rate=0.96)
#     # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#     learning_rate = MultiStepWarmUpLR(
#         initial_learning_rate=cfg['init_lr'],
#         lr_steps=[e * steps_per_epoch for e in cfg['lr_decay_epoch']],
#         lr_rate=cfg['lr_rate'],
#         warmup_steps=cfg['warmup_epoch'] * steps_per_epoch,
#         min_lr=cfg['min_lr'])

#     optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=cfg['momentum'], nesterov=True)

#     multi_loss = MultiBoxLoss(num_class=len(label_classes), neg_pos_ratio=3)

#     train_log_dir = 'logs/train'
#     train_summary_writer = tf.summary.create_file_writer(train_log_dir)

#     @tf.function
#     def train_step(inputs, labels):
#         with tf.GradientTape() as tape:
#             predictions = model(inputs, training=True)
#             losses = {}
#             losses['reg'] = tf.reduce_sum(model.losses)  # unused. Init for redefine network
#             losses['loc'], losses['class'] = multi_loss(labels, predictions)
#             total_loss = tf.add_n([l for l in losses.values()])

#         grads = tape.gradient(total_loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))

#         return total_loss, losses

#     for epoch in range(init_epoch + 1, cfg['epoch']):
#         try:
#             start = time.time()
#             avg_loss = 0.0
#             for step, (inputs, labels) in enumerate(train_dataset.take(steps_per_epoch)):

#                 load_t0 = time.time()
#                 total_loss, losses = train_step(inputs, labels)
#                 avg_loss = (avg_loss * step + total_loss.numpy()) / (step + 1)
#                 load_t1 = time.time()
#                 batch_time = load_t1 - load_t0

#                 steps = steps_per_epoch * epoch + step
#                 with train_summary_writer.as_default():
#                     tf.summary.scalar('loss/total_loss', total_loss, step=steps)
#                     for k, l in losses.items():
#                         tf.summary.scalar('loss/{}'.format(k), l, step=steps)
#                     tf.summary.scalar('learning_rate', optimizer.lr(steps), step=steps)

#                 print(
#                     f"\rEpoch: {epoch + 1}/{cfg['epoch']} | Batch {step + 1}/{steps_per_epoch} | Batch time {batch_time:.3f} || Loss: {total_loss:.6f} | loc loss:{losses['loc']:.6f} | class loss:{losses['class']:.6f} ",
#                     end='', flush=True)

#             print(
#                 f"\nEpoch: {epoch + 1}/{cfg['epoch']}  | Epoch time {(load_t1 - start):.3f} || Average Loss: {avg_loss:.6f}")

#             with train_summary_writer.as_default():
#                 tf.summary.scalar('loss/avg_loss', avg_loss, step=epoch)

#             if (epoch + 1) % cfg['save_freq'] == 0:
#                 filepath = os.path.join(weights_dir, f'weights_epoch_{(epoch + 1):03d}.h5')
#                 model.save_weights(filepath)
#                 if os.path.exists(filepath):
#                     print(f">>>>>>>>>>Save weights file at {filepath}<<<<<<<<<<")

#         except KeyboardInterrupt:
#             print('interrupted')
#             # filepath = os.path.join(weights_dir, 'weights_last.h5')
#             # model.save_weights(filepath)
#             # print(f'model saved into: {filepath}')
#             exit(0)


# if __name__ == '__main__':
#     try:
#         app.run(main)
#     except SystemExit:
#         pass