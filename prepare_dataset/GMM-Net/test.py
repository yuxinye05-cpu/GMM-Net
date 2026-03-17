import joblib,copy
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader
import torch,sys
from tqdm import tqdm

from collections import OrderedDict
from lib.visualize import save_img,group_images,concat_result,preprocessed_image_show
import os
import argparse
from lib.logger import Logger, Print_Logger
from lib.extract_patches import *
from os.path import join
from lib.dataset import TestDataset
from lib.metrics import Evaluate
import models
from lib.common import setpu_seed,dict_round
from config import parse_args
from lib.pre_processing import my_PreProc,preprocessed_image

setpu_seed(2021)

class Test():
    def __init__(self, args):
        # 采用图像分块的方式测试
        self.args = args
        assert (args.stride_height <= args.test_patch_height and args.stride_width <= args.test_patch_width)
        # save path
        self.path_experiment = args.outf

        self.patches_imgs_test, self.test_imgs, self.test_masks, self.test_FOVs, self.new_height, self.new_width = get_data_test_overlap(
            test_data_path_list=args.test_data_path_list,
            patch_height=args.test_patch_height,
            patch_width=args.test_patch_width,
            stride_height=args.stride_height,
            stride_width=args.stride_width
        )
        self.img_height = self.test_imgs.shape[2]
        self.img_width = self.test_imgs.shape[3]

        test_set = TestDataset(self.patches_imgs_test)
        self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Inference prediction process
    def inference(self, net):
        net.eval()
        preds = []
        device = next(net.parameters()).device  # 获取模型所在显卡
        with torch.no_grad():
            for batch_idx, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                inputs = inputs.to(device)  # ✅ 移动到模型所在设备
                outputs = net(inputs)
                outputs = outputs[:, 1].data.cpu().numpy()
                preds.append(outputs)
        predictions = np.concatenate(preds, axis=0)
        self.pred_patches = np.expand_dims(predictions, axis=1)

    # Evaluate ate and visualize the predicted images
    def evaluate(self):
        # 对图片补丁进行拼接，使其变成一整张图片
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        ## restore to original dimensions 将图片恢复到原始尺寸
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]

        #predictions only inside the FOV
        y_scores, y_true = pred_only_in_FOV(self.pred_imgs, self.test_masks, self.test_FOVs)
        eval = Evaluate(save_path=self.path_experiment)
        eval.add_batch(y_true, y_scores)
        log = eval.save_all_result(plot_curve=True,save_name="performance.txt")
        # save labels and probs for plot ROC and PR curve when k-fold Cross-validation
        np.save('{}/result.npy'.format(self.path_experiment), np.asarray([y_true, y_scores]))
        return dict_round(log, 6)

    # save segmentation imgs
    def save_segmentation_result(self):
        img_path_list, _, _ = load_file_path_txt(self.args.test_data_path_list)
        img_name_list = [item.split('/')[-1].split('.')[0] for item in img_path_list]

        kill_border(self.pred_imgs, self.test_FOVs) # only for visualization
        self.test_imgs=kill_border(self.test_imgs, self.test_FOVs)# 新加的去除边界多余像素代码
        self.save_img_path = join(self.path_experiment,'result_img')
        if not os.path.exists(join(self.save_img_path)):
            os.makedirs(self.save_img_path)
        #self.test_imgs = my_PreProc(self.test_imgs) # Uncomment to save the pre processed image

        for i in range(self.test_imgs.shape[0]):
            # ================= 原有：拼接四图 =================
            total_img = concat_result(
                self.test_imgs[i],
                self.pred_imgs[i],
                self.test_masks[i]
            )
            save_img(
                total_img,
                join(self.save_img_path, "Result_" + img_name_list[i] + '.png')
            )

            # ================= 1. 保存预测概率图 =================
            prob_map = self.pred_imgs[i][0]  # (H, W)
            prob_map = (prob_map * 255).astype(np.uint8)
            prob_map = prob_map[:, :, np.newaxis]  # (H, W, 1)

            prob_name = f"test_{i + 1:02d}.png"
            save_img(prob_map, join(self.save_img_path, prob_name))

            # ================= 2. 保存原始图像 =================
            origin = self.test_imgs[i][0]  # (H, W)
            origin = (origin * 255).astype(np.uint8)
            origin = origin[:, :, np.newaxis]  # (H, W, 1)

            origin_name = f"origin_{i + 1:02d}.png"
            save_img(origin, join(self.save_img_path, origin_name))

            # ================= 3. 保存 Ground Truth =================
            gt = self.test_masks[i][0]  # (H, W)
            gt = (gt * 255).astype(np.uint8)
            gt = gt[:, :, np.newaxis]  # (H, W, 1)

            gt_name = f"GT_{i + 1:02d}.png"
            save_img(gt, join(self.save_img_path, gt_name))

        # for i in range(self.test_imgs.shape[0]):
        #     total_img = concat_result(self.test_imgs[i],self.pred_imgs[i],self.test_masks[i])
        #     save_img(total_img,join(self.save_img_path, "Result_"+img_name_list[i]+'.png'))
        #     # -------- 新增：单独保存预测概率图 --------
        #     prob_map = self.pred_imgs[i][0]  # (H, W)
        #     prob_map = prob_map[:, :, np.newaxis]  # (H, W, 1)
        #     prob_map = (prob_map * 255).astype(np.uint8)
        #
        #     prob_name = f"test_{i + 1:02d}.png"
        #     save_img(prob_map, join(self.save_img_path, prob_name))

        x0,x1,x2,x3,x4=preprocessed_image(self.test_imgs)
        preprocessed_image_show(x0,x1,x2,x3,x4,self.path_experiment)



    # Val on the test set at each epoch
    def val(self):
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        ## recover to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]

        #predictions only inside the FOV
        y_scores, y_true = pred_only_in_FOV(self.pred_imgs, self.test_masks, self.test_FOVs)
        eval = Evaluate(save_path=self.path_experiment)
        eval.add_batch(y_true, y_scores)
        confusion,accuracy,specificity,sensitivity,precision = eval.confusion_matrix()
        log = OrderedDict([('val_auc_roc', eval.auc_roc()),
                           ('val_f1', eval.f1_score()),
                           ('val_acc', accuracy),
                           ('SE', sensitivity),
                           ('SP', specificity),
                           ('PR',precision)])
        eval.save_all_result(plot_curve=True, save_name="val_metrics.txt")
        return dict_round(log, 6)

if __name__ == '__main__':
    args = parse_args()
    save_path = args.outf
    sys.stdout = Print_Logger(os.path.join(save_path, 'test_log.txt'))
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    #net = models.UNetFamily.Dense_Unet(1,2).to(device)
    #net = models.UNetFamily.U_Net(1,2).to(device)
    #net = models.GMMNet(num_classes=2).to(device)
    net=models.Test().to(device)
    #net = nn.DataParallel(net)
    cudnn.benchmark = True

    # Load checkpoint
    print('==> Loading checkpoint...')
    checkpoint = torch.load(join(save_path, 'best_model.pth'))
    net.load_state_dict(checkpoint['net'])

    eval = Test(args)
    eval.inference(net)
    print(eval.evaluate())
    eval.save_segmentation_result()
