import cv2
import time
import progressbar
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from networks.mvsnet import MVSNet
from datasets import get_loader
from tools import *
from loss import mvs_loss
from datasets.data_io import save_pfm, read_pfm
from filter import gipuma_filter, pcd_filter, dypcd_filter
from filter.tank_test_config import tank_cfg


class Model:
    def __init__(self, args):

        if args.vis:
            self.args = args
            return

        cudnn.benchmark = True

        init_distributed_mode(args)

        self.args = args
        self.device = torch.device("cpu" if self.args.no_cuda or not torch.cuda.is_available() else "cuda")

        self.network = MVSNet(ndepths=args.ndepths, depth_interval_ratio=args.interval_ratio, fea_mode=args.fea_mode,
                              agg_mode=args.agg_mode, depth_mode=args.depth_mode).to(self.device)

        if self.args.distributed and self.args.sync_bn:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)

        if not (self.args.val or self.args.test):

            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.network.parameters()), lr=args.lr,
                                              weight_decay=args.wd)
            self.lr_scheduler = get_schedular(self.optimizer, self.args)
            self.train_loader, self.train_sampler = get_loader(args, args.datapath, args.trainlist, args.nviews, "train")

        if not self.args.test:
            self.loss_func = mvs_loss
            self.val_loader, self.val_sampler = get_loader(args, args.datapath, args.testlist, 5, "test")
            if is_main_process():
                self.writer = SummaryWriter(log_dir=args.log_dir, comment="Record network info")

        self.network_without_ddp = self.network
        if self.args.distributed:
            self.network = DistributedDataParallel(self.network, device_ids=[self.args.local_rank])
            self.network_without_ddp = self.network.module

        if self.args.resume:
            checkpoint = torch.load(self.args.resume, map_location="cpu")
            if not (self.args.val or self.args.test or self.args.blendedmvs_finetune):
                self.args.start_epoch = checkpoint["epoch"] + 1
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.network_without_ddp.load_state_dict(checkpoint["model"])

    def main(self):
        if self.args.vis:
            self.visualization()
            return
        if self.args.val:
            self.validate()
            return
        if self.args.test:
            self.test()
            return
        self.train()

    def train(self):
        for epoch in range(self.args.start_epoch, self.args.start_epoch + self.args.epochs):
            if self.args.distributed:
                self.train_sampler.set_epoch(epoch)
            self.train_epoch(epoch)
            if is_main_process():
                torch.save({
                    'epoch': epoch,
                    'model': self.network_without_ddp.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict()},
                    "{}/model_{:0>6}.ckpt".format(self.args.log_dir, epoch))

            if (epoch % self.args.eval_freq == 0) or (epoch == self.args.epochs - 1):
                self.validate(epoch)
            torch.cuda.empty_cache()

    def train_epoch(self, epoch):
        self.network.train()

        if is_main_process():
            pwidgets = [progressbar.Percentage(), " ", progressbar.Counter(format='%(value)02d/%(max_value)d'), " ", progressbar.Bar(), " ",
                        progressbar.Timer(), ",", progressbar.ETA(), ",", progressbar.Variable('LR', width=1), ",",
                        progressbar.Variable('Loss', width=1), ",", progressbar.Variable('Th2', width=1), ",",
                        progressbar.Variable('Th4', width=1), ",", progressbar.Variable('Th8', width=1)]
            pbar = progressbar.ProgressBar(widgets=pwidgets, max_value=len(self.train_loader),
                                           prefix="Epoch {}/{}: ".format(epoch, self.args.epochs)).start()

        avg_scalars = DictAverageMeter()

        for batch, data in enumerate(self.train_loader):
            data = tocuda(data)

            outputs = self.network(data["imgs"], data["proj_matrices"], data["depth_values"])

            loss = self.loss_func(outputs, data["depth"], data["mask"], self.args.depth_mode, dlossw=self.args.dlossw)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.lr_scheduler.step(epoch + batch / len(self.train_loader))

            gt_depth = data["depth"]["stage{}".format(len(self.args.ndepths))]
            mask = data["mask"]["stage{}".format(len(self.args.ndepths))]
            thres2mm = Thres_metrics(outputs["depth"], gt_depth, mask > 0.5, 2)
            thres4mm = Thres_metrics(outputs["depth"], gt_depth, mask > 0.5, 4)
            thres8mm = Thres_metrics(outputs["depth"], gt_depth, mask > 0.5, 8)
            abs_depth_error = AbsDepthError_metrics(outputs["depth"], gt_depth, mask > 0.5)

            scalar_outputs = {"loss": loss,
                              "abs_depth_error": abs_depth_error,
                              "thres2mm_error": thres2mm,
                              "thres4mm_error": thres4mm,
                              "thres8mm_error": thres8mm}

            image_outputs = {"depth_est": outputs["depth"] * mask,
                             "depth_est_nomask": outputs["depth"],
                             "depth_gt": gt_depth,
                             "ref_img": data["imgs"][:, 0],
                             "mask": mask,
                             "errormap": (outputs["depth"] - gt_depth).abs() * mask,
                             }

            if self.args.distributed:
                scalar_outputs = reduce_scalar_outputs(scalar_outputs)

            scalar_outputs, image_outputs = tensor2float(scalar_outputs), tensor2numpy(image_outputs)

            if is_main_process():
                avg_scalars.update(scalar_outputs)
                if batch >= len(self.train_loader) - 1:
                    save_scalars(self.writer, 'train_avg', avg_scalars.avg_data, epoch)
                if (epoch * len(self.train_loader) + batch) % self.args.summary_freq == 0:
                    save_scalars(self.writer, 'train', scalar_outputs, epoch * len(self.train_loader) + batch)
                    save_images(self.writer, 'train', image_outputs, epoch * len(self.train_loader) + batch)

                pbar.update(batch, LR=self.optimizer.param_groups[0]['lr'],
                            Loss="{:.3f}|{:.3f}".format(scalar_outputs["loss"], avg_scalars.avg_data["loss"]),
                            Th2="{:.3f}|{:.3f}".format(scalar_outputs["thres2mm_error"], avg_scalars.avg_data["thres2mm_error"]),
                            Th4="{:.3f}|{:.3f}".format(scalar_outputs["thres4mm_error"], avg_scalars.avg_data["thres4mm_error"]),
                            Th8="{:.3f}|{:.3f}".format(scalar_outputs["thres8mm_error"], avg_scalars.avg_data["thres8mm_error"]))

        if is_main_process():
            pbar.finish()

    @torch.no_grad()
    def validate(self, epoch=0):
        self.network.eval()

        if is_main_process():
            pwidgets = [progressbar.Percentage(), " ", progressbar.Counter(format='%(value)02d/%(max_value)d'), " ", progressbar.Bar(), " ",
                        progressbar.Timer(), ",", progressbar.ETA(), ",", progressbar.Variable('Loss', width=1), ",",
                        progressbar.Variable('Th2', width=1), ",", progressbar.Variable('Th4', width=1), ",",
                        progressbar.Variable('Th8', width=1)]
            pbar = progressbar.ProgressBar(widgets=pwidgets, max_value=len(self.val_loader), prefix="Val:").start()

        avg_scalars = DictAverageMeter()

        for batch, data in enumerate(self.val_loader):
            data = tocuda(data)

            outputs = self.network(data["imgs"], data["proj_matrices"], data["depth_values"])

            loss = self.loss_func(outputs, data["depth"], data["mask"], self.args.depth_mode, dlossw=self.args.dlossw)

            gt_depth = data["depth"]["stage{}".format(len(self.args.ndepths))]
            mask = data["mask"]["stage{}".format(len(self.args.ndepths))]
            thres2mm = Thres_metrics(outputs["depth"], gt_depth, mask > 0.5, 2)
            thres4mm = Thres_metrics(outputs["depth"], gt_depth, mask > 0.5, 4)
            thres8mm = Thres_metrics(outputs["depth"], gt_depth, mask > 0.5, 8)
            abs_depth_error = AbsDepthError_metrics(outputs["depth"], gt_depth, mask > 0.5)

            scalar_outputs = {"loss": loss,
                              "abs_depth_error": abs_depth_error,
                              "thres2mm_error": thres2mm,
                              "thres4mm_error": thres4mm,
                              "thres8mm_error": thres8mm}

            image_outputs = {"depth_est": outputs["depth"] * mask,
                             "depth_est_nomask": outputs["depth"],
                             "depth_gt": gt_depth,
                             "ref_img": data["imgs"][:, 0],
                             "mask": mask,
                             "errormap": (outputs["depth"] - gt_depth).abs() * mask,
                             }

            if self.args.distributed:
                scalar_outputs = reduce_scalar_outputs(scalar_outputs)

            scalar_outputs, image_outputs = tensor2float(scalar_outputs), tensor2numpy(image_outputs)

            if is_main_process():
                avg_scalars.update(scalar_outputs)
                if batch >= len(self.val_loader) - 1:
                    save_scalars(self.writer, 'test_avg', avg_scalars.avg_data, epoch)
                if (epoch * len(self.val_loader) + batch) % self.args.summary_freq == 0:
                    save_scalars(self.writer, 'test', scalar_outputs, epoch * len(self.val_loader) + batch)
                    save_images(self.writer, 'test', image_outputs, epoch * len(self.val_loader) + batch)

                pbar.update(batch,
                            Loss="{:.3f}|{:.3f}".format(scalar_outputs["loss"], avg_scalars.avg_data["loss"]),
                            Th2="{:.3f}|{:.3f}".format(scalar_outputs["thres2mm_error"], avg_scalars.avg_data["thres2mm_error"]),
                            Th4="{:.3f}|{:.3f}".format(scalar_outputs["thres4mm_error"], avg_scalars.avg_data["thres4mm_error"]),
                            Th8="{:.3f}|{:.3f}".format(scalar_outputs["thres8mm_error"], avg_scalars.avg_data["thres8mm_error"]))

        if is_main_process():
            pbar.finish()

    @torch.no_grad()
    def test(self):
        self.network.eval()

        if self.args.testpath_single_scene:
            self.args.datapath = os.path.dirname(self.args.testpath_single_scene)

        if self.args.testlist != "all":
            with open(self.args.testlist) as f:
                content = f.readlines()
                testlist = [line.rstrip() for line in content]
        else:
            # for tanks & temples or eth3d or colmap
            testlist = [e for e in os.listdir(self.args.datapath) if os.path.isdir(os.path.join(self.args.datapath, e))] \
                if not self.args.testpath_single_scene else [os.path.basename(self.args.testpath_single_scene)]

        num_stage = len(self.args.ndepths)

        # step1. save all the depth maps and the masks in outputs directory
        for scene in testlist:

            if scene in tank_cfg.scenes:
                scene_cfg = getattr(tank_cfg, scene)
                self.args.max_h = scene_cfg.max_h
                self.args.max_w = scene_cfg.max_w

            TestImgLoader, _ = get_loader(self.args, self.args.datapath, [scene], self.args.num_view, mode="test")

            for batch_idx, sample in enumerate(TestImgLoader):
                sample_cuda = tocuda(sample)
                start_time = time.time()
                outputs = self.network(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
                end_time = time.time()
                outputs = tensor2numpy(outputs)
                del sample_cuda
                filenames = sample["filename"]
                cams = sample["proj_matrices"]["stage{}".format(num_stage)].numpy()
                imgs = sample["imgs"].numpy()
                print('Iter {}/{}, Time:{} Res:{}'.format(batch_idx, len(TestImgLoader), end_time - start_time, imgs[0].shape))

                # save depth maps and confidence maps
                for filename, cam, img, depth_est, depth2, depth1, photometric_confidence, pc2, pc1 \
                        in zip(filenames, cams, imgs, outputs["depth"],
                               outputs["stage2"]["depth"],
                               outputs["stage1"]["depth"],
                               outputs["photometric_confidence"],
                               outputs["stage2"]["photometric_confidence"],
                               outputs["stage1"]["photometric_confidence"]):

                    depth_filename2 = os.path.join(self.args.outdir, filename.format('depth_est', '_stage2.pfm'))
                    depth_filename1 = os.path.join(self.args.outdir, filename.format('depth_est', '_stage1.pfm'))

                    h, w = photometric_confidence.shape
                    pc2 = cv2.resize(pc2, (w, h), interpolation=cv2.INTER_NEAREST)
                    pc1 = cv2.resize(pc1, (w, h), interpolation=cv2.INTER_NEAREST)
                    confidence_filename2 = os.path.join(self.args.outdir, filename.format('confidence', '_stage2.pfm'))
                    confidence_filename1 = os.path.join(self.args.outdir, filename.format('confidence', '_stage1.pfm'))

                    img = img[0]  # ref view
                    cam = cam[0]  # ref cam
                    depth_filename = os.path.join(self.args.outdir, filename.format('depth_est', '.pfm'))
                    confidence_filename = os.path.join(self.args.outdir, filename.format('confidence', '.pfm'))
                    cam_filename = os.path.join(self.args.outdir, filename.format('cams', '_cam.txt'))
                    img_filename = os.path.join(self.args.outdir, filename.format('images', '.jpg'))
                    # ply_filename = os.path.join(self.args.outdir, filename.format('ply_local', '.ply'))
                    os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
                    # os.makedirs(ply_filename.rsplit('/', 1)[0], exist_ok=True)
                    # save depth maps
                    save_pfm(depth_filename, depth_est)
                    save_pfm(depth_filename2, depth2)
                    save_pfm(depth_filename1, depth1)
                    # save confidence maps
                    save_pfm(confidence_filename, photometric_confidence)
                    save_pfm(confidence_filename2, pc2)
                    save_pfm(confidence_filename1, pc1)
                    # save cams, img
                    write_cam(cam_filename, cam)
                    img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(img_filename, img_bgr)

                    # if num_stage == 1:
                    #     downsample_img = cv2.resize(img, (int(img.shape[1] * 0.25), int(img.shape[0] * 0.25)))
                    # elif num_stage == 2:
                    #     downsample_img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))
                    # elif num_stage == 3:
                    #     downsample_img = img
                    #
                    # if batch_idx % self.args.save_freq == 0:
                    #     generate_pointcloud(downsample_img, depth_est, ply_filename, cam[1, :3, :3])

            torch.cuda.empty_cache()

        # step2. filter saved depth maps with photometric confidence maps and geometric constraints
        if self.args.filter_method == "pcd":
            # support multi-processing, the default number of worker is 4
            pcd_filter(self.args, testlist, self.args.num_worker)
        elif self.args.filter_method == "dypcd":
            dypcd_filter(self.args, testlist, self.args.num_worker)
        else:
            gipuma_filter(testlist, self.args.outdir, self.args.prob_threshold, self.args.disp_threshold, self.args.num_consistent,
                          self.args.fusibile_exe_path)

    @torch.no_grad()
    def visualization(self):

        import matplotlib as mpl
        import matplotlib.cm as cm
        from PIL import Image

        save_dir = self.args.depth_img_save_dir
        depth_path = self.args.depth_path

        depth, scale = read_pfm(depth_path)
        vmax = np.percentile(depth, 95)
        normalizer = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
        im = Image.fromarray(colormapped_im)
        im.save(os.path.join(save_dir, "depth.png"))

        print("Successfully visualize!")

