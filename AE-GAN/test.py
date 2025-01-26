import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from options.config import load_config
from data import create_dataset
from models import create_model
from util import util
import numpy as np
from matplotlib import pyplot as plt
def show_image(imgs, fname=None, cmap='gray', norm=False, vmin=0, vmax=1, transpose='z', origin='lower'):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    imgs = imgs.detach().cpu().numpy()
    # imgs = imgs.detach().numpy()
    for i, ax in zip(range(0, imgs.shape[0], imgs.shape[0] // 16), axes):
        ax.imshow(imgs[i], cmap=plt.get_cmap(cmap), aspect='equal', origin=origin)
    if fname:
        fig.savefig(fname)
        plt.close(fig)
    else:
        return fig

if __name__ == '__main__':
    opt = load_config()
    opt.load_size = 128
    opt.results_dir = 'results/'
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.isTrain = False
    phase = 'val' if not opt.isTrain else 'train'
    dataloader = create_dataset(opt)
    opt.n_input_modal = dataloader.dataset.n_modal - 1
    opt.modal_names = dataloader.dataset.get_modal_names()
    n_modal = 1 if 'encoder' in opt.name or 'pix2pix' in opt.name or 'cycle' in opt.name else dataloader.dataset.n_modal
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    dst_dir = os.path.join(opt.results_dir, opt.name, phase + '-' + str(opt.epoch))
    os.makedirs(dst_dir, exist_ok=True)

    label_counters = {}  # 用于存储每个标签的计数器

    for i, data in enumerate(dataloader):
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference: forward() and compute_visuals()
        visuals = model.get_current_visuals()  # get image results

        for label, image in visuals.items():
            # 初始化标签计数器，如果不存在的话
            if label not in label_counters:
                label_counters[label] = 0

            # 获取当前标签的计数器值
            counter_value = label_counters[label]

            # 更新计数器
            label_counters[label] += 1

            # 构建保存文件名，添加计数器后缀
            filename = f"{label}_{counter_value}.png"

            # 获取图像数据并保存
            image_numpy = np.squeeze(image)
            show_image(image_numpy, os.path.join('D:/Code/AE-GAN/test/', filename))

        #     image_numpy = util.tensor2im(image)
        #     imgs.append(image_numpy)
        #     labels.append(label)
        #     label_dst_dir = os.path.join(dst_dir, label)
        #     os.makedirs(label_dst_dir, exist_ok=True)
        #     util.save_image(image_numpy, os.path.join(label_dst_dir, '{}.jpg'.format(i//n_modal + 1)))
        # cat_img = np.concatenate(imgs, axis=1)
        # cat_dir = os.path.join(dst_dir, '-'.join(labels))
        # os.makedirs(cat_dir, exist_ok=True)
        # util.save_image(cat_img, os.path.join(cat_dir, '{}.jpg'.format(i//n_modal+1)))