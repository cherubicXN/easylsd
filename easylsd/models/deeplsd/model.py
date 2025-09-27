"""
Regress the distance function map to all the line segments of an image.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .line_utils import merge_lines, filter_outlier_lines
from .tensor_utils import preprocess_angle
try:
    from pytlsd import lsd
except Exception:
    lsd = None

#Base model, copy from DeepLSD repo
from abc import ABCMeta, abstractmethod
from omegaconf import OmegaConf
from copy import copy


class VGGUNet(torch.nn.Module):
    def __init__(self, tiny=False):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        if tiny:
            sizes = [32, 64, 128, 256]
        else:
            sizes = [64, 128, 256, 512]
        
        # Encoder blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(1, sizes[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[0]),
            nn.Conv2d(sizes[0], sizes[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[0]),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(sizes[0], sizes[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[1]),
            nn.Conv2d(sizes[1], sizes[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[1]),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(sizes[1], sizes[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[2]),
            nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[2]),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(sizes[2], sizes[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[3]),
            nn.Conv2d(sizes[3], sizes[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[3]),
        )

        # Decoder blocks
        self.deblock4 = nn.Sequential(
            nn.Conv2d(sizes[3], sizes[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[2]),
            nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[2]),
        )
        self.deblock3 = nn.Sequential(
            nn.Conv2d(sizes[3], sizes[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[2]),
            nn.Conv2d(sizes[2], sizes[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[1]),
        )
        self.deblock2 = nn.Sequential(
            nn.Conv2d(sizes[2], sizes[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[1]),
            nn.Conv2d(sizes[1], sizes[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[0]),
        )
        self.deblock1 = nn.Sequential(
            nn.Conv2d(sizes[1], sizes[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[0]),
            nn.Conv2d(sizes[0], sizes[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[0]),
        )

    def forward(self, inputs):
        # Encoding
        features = [self.block1(inputs)]
        for block in [self.block2, self.block3, self.block4]:
            features.append(block(self.pool(features[-1])))

        # Decoding
        out = self.deblock4(features[-1])
        for deblock, feat in zip(
            [self.deblock3, self.deblock2, self.deblock1], features[:-1][::-1]):
            out = deblock(torch.cat([
                F.interpolate(out, feat.shape[2:4], mode='bilinear'),
                feat], dim=1))

        return out  # dim = 32 if tiny else 64


class BaseModel(nn.Module, metaclass=ABCMeta):
    """
    What the child model is expect to declare:
        default_conf: dictionary of the default configuration of the model.
        It overwrites base_default_conf in BaseModel, and it is overwritten by
        the user-provided configuration passed to __init__.
        Configurations can be nested.

        required_data_keys: list of expected keys in the input data dictionary.

        strict_conf (optional): boolean. If false, BaseModel does not raise
        an error when the user provides an unknown configuration entry.

        _init(self, conf): initialization method, where conf is the final
        configuration object (also accessible with `self.conf`). Accessing
        unkown configuration entries will raise an error.

        _forward(self, data): method that returns a dictionary of batched
        prediction tensors based on a dictionary of batched input data tensors.

        loss(self, pred, data): method that returns a dictionary of losses,
        computed from model predictions and input data. Each loss is a batch
        of scalars, i.e. a torch.Tensor of shape (B,).
        The total loss to be optimized has the key `'total'`.

        metrics(self, pred, data): method that returns a dictionary of metrics,
        each as a batch of scalars.
    """
    base_default_conf = {
        'name': None,
        'trainable': False,
    }
    default_conf = {}
    required_data_keys = []
    strict_conf = False

    def __init__(self, conf):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        default_conf = OmegaConf.merge(
                OmegaConf.create(self.base_default_conf),
                OmegaConf.create(self.default_conf))
        if self.strict_conf:
            OmegaConf.set_struct(default_conf, False)
        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = conf = OmegaConf.merge(default_conf, conf)
        OmegaConf.set_readonly(conf, True)
        self.required_data_keys = copy(self.required_data_keys)
        self._init(conf)

        if not conf.trainable:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, data):
        """Check the data and call the _forward method of the child model."""
        for key in self.required_data_keys:
            assert key in data, 'Missing key {} in data'.format(key)
        return self._forward(data)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def loss(self, pred, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def metrics(self, pred, data):
        """To be implemented by the child class."""
        raise NotImplementedError


class DeepLSD(BaseModel):
    default_conf = {
        'tiny': False,
        'sharpen': True,
        'line_neighborhood': 5,
        'loss_weights': {
            'df': 1.,
            'angle': 1.,
        },
        'multiscale': False,
        'scale_factors': [1., 1.5],
        'detect_lines': False,
        'line_detection_params': {
            'merge': False,
            'grad_nfa': True,
            'optimize': False,
            'use_vps': False,
            'optimize_vps': False,
            'filtering': 'normal',
            'grad_thresh': 3,
            'lambda_df': 1.,
            'lambda_grad': 1.,
            'lambda_vp': 0.5,
        },
    }
    required_data_keys = ['image']

    def _init(self, conf):
        # Base network
        self.backbone = VGGUNet(tiny=self.conf.tiny)
        dim = 32 if self.conf.tiny else 64

        # Predict the distance field and angle to the nearest line
        # DF head
        self.df_head = nn.Sequential(
            nn.Conv2d(dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(),
        )

        # Closest line direction head
        self.angle_head = nn.Sequential(
            nn.Conv2d(dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Loss
        self.l1_loss_fn = nn.L1Loss(reduction='none')
        self.l2_loss_fn = nn.MSELoss(reduction='none')

    def normalize_df(self, df):
        return -torch.log(df / self.conf.line_neighborhood + 1e-6)

    def denormalize_df(self, df_norm):
        return torch.exp(-df_norm) * self.conf.line_neighborhood

    def _forward(self, data):
        outputs = {}

        if self.conf.multiscale:
            outputs = self.ms_forward(data)
        else:
            base = self.backbone(data['image'])

            # DF prediction
            if self.conf.sharpen:
                outputs['df_norm'] = self.df_head(base).squeeze(1)
                outputs['df'] = self.denormalize_df(outputs['df_norm'])
            else:
                outputs['df'] = self.df_head(base).squeeze(1)

            # Closest line direction prediction
            outputs['line_level'] = self.angle_head(base).squeeze(1) * np.pi

        # Detect line segments
        if self.conf.detect_lines:
            lines = []
            np_img = (data['image'].cpu().numpy()[:, 0] * 255).astype(np.uint8)
            np_df = outputs['df'].cpu().numpy()
            np_ll = outputs['line_level'].cpu().numpy()
            vps, vp_labels = [], []
            for img, df, ll in zip(np_img, np_df, np_ll):
                line, label, vp = self.detect_afm_lines(
                    img, df, ll, **self.conf.line_detection_params)
                lines.append(line)
                vp_labels.append(label)
                vps.append(vp)
            outputs['vp_labels'] = vp_labels
            outputs['vps'] = vps
            outputs['lines'] = lines

        return outputs

    def ms_forward(self, data):
        """ Do several forward passes at multiple image resolutions
            and aggregate the results before extracting the lines. """
        img_size = data['image'].shape[2:]

        # Forward pass for each scale
        pred_df, pred_angle = [], []
        for s in self.conf.scale_factors:
            img = F.interpolate(data['image'], scale_factor=s, mode='bilinear')
            with torch.no_grad():
                base = self.backbone(img)
                if self.conf.sharpen:
                    pred_df.append(self.denormalize_df(self.df_head(base)))
                else:
                    pred_df.append(self.df_head(base))
                pred_angle.append(self.angle_head(base) * np.pi)

        # Fuse the outputs together
        for i in range(len(self.conf.scale_factors)):
            pred_df[i] = F.interpolate(pred_df[i], img_size,
                                       mode='bilinear').squeeze(1)
            pred_angle[i] = F.interpolate(pred_angle[i], img_size,
                                          mode='nearest').squeeze(1)
        fused_df = torch.stack(pred_df, dim=0).mean(dim=0)
        fused_angle = torch.median(torch.stack(pred_angle, dim=0), dim=0)[0]

        out = {'df': fused_df, 'line_level': fused_angle}
        return out

    def detect_afm_lines(
        self, img, df, line_level, optimize=False, use_vps=False,
        optimize_vps=False, filtering='normal', merge=False,
        grad_thresh=3, lambda_df=1., lambda_grad=1., lambda_vp=0.5,
        grad_nfa=True):
        """ Detect lines from the offset field and potentially the line angle.
            Offer the possibility to ignore line in high DF values, to merge
            close-by lines and to optimize them to better fit the DF + angle. """
        gradnorm = np.maximum(5 - df, 0).astype(np.float64)
        angle = line_level.astype(np.float64) - np.pi / 2
        angle = preprocess_angle(angle, img, mask=True)[0]
        angle[gradnorm < grad_thresh] = -1024
        lines = lsd(
            img.astype(np.float64), scale=1., gradnorm=gradnorm,
            gradangle=angle, grad_nfa=grad_nfa)[:, :4].reshape(-1, 2, 2)

        # Optionally filter out lines based on the DF and line_level
        if filtering:
            if filtering == 'strict':
                df_thresh, ang_thresh = 1., np.pi / 12
            else:
                df_thresh, ang_thresh = 1.5, np.pi / 9
            angle = line_level - np.pi / 2
            lines = filter_outlier_lines(
                img, lines[:, :, [1, 0]], df, angle, mode='inlier_thresh',
                use_grad=False, inlier_thresh=0.5, df_thresh=df_thresh,
                ang_thresh=ang_thresh)[0][:, :, [1, 0]]

        # Optimize the lines with respect to the DF and line level
        vps = np.array([])
        vp_labels = np.array([-1] * len(lines))
        if optimize:
            if merge:
                lines = merge_lines(lines, thresh=4,
                                    overlap_thresh=0).astype(np.float32)

            rows, cols = df.shape
            angle, _ = preprocess_angle(
                line_level - np.pi / 2, img)
            orientations = get_line_orientation(lines[:, :, [1, 0]],
                                                angle)[:, None]
            oriented_lines = np.concatenate([lines.reshape(-1, 4),
                                             orientations], axis=1)
            lines, vp_labels, vps = line_optim(
                oriented_lines, (df ** (1/4)).flatten(), angle.flatten(),
                rows, cols, use_vps, optimize_vps,
                lambda_df, lambda_grad, lambda_vp)
            lines = np.array(lines).reshape(-1, 2, 2).astype(np.float32)

        # Merge close-by lines together
        if merge and not optimize:
            lines = merge_lines(lines, thresh=4,
                                overlap_thresh=0).astype(np.float32)

        return lines, vp_labels, vps

    def ha(self, data, num_H=10, aggregation='median'):
        """ Perform homography augmentation at test time on a single image. """
        df, line_level, _ = torch_homography_adaptation(
            data['image'], self, num_H, aggregation=aggregation)
        outputs = {'df': df, 'line_level': line_level}

        # Detect line segments
        if self.conf.detect_lines:
            np_img = (data['image'].cpu().numpy()[0, 0] * 255).astype(np.uint8)
            np_df = df.cpu().numpy()
            np_ll = line_level.cpu().numpy()
            outputs['lines'] = self.detect_afm_lines(
                np_img, np_df, np_ll, **self.conf.line_detection_params)[0]

        return outputs

    def loss(self, pred, data):
        outputs = {}
        loss = 0

        # Retrieve the mask of valid pixels
        valid_mask = data['ref_valid_mask']
        valid_norm = valid_mask.sum(dim=[1, 2])
        valid_norm[valid_norm == 0] = 1

        # Retrieve the mask of pixels close to GT lines
        line_mask = (valid_mask
                     * (data['df'] < self.conf.line_neighborhood).float())
        line_norm = line_mask.sum(dim=[1, 2])
        line_norm[line_norm == 0] = 1

        # DF loss, with supervision only on the lines neighborhood
        if self.conf.sharpen:
            df_loss = self.l1_loss_fn(pred['df_norm'],
                                      self.normalize_df(data['df']))
        else:
            df_loss = self.l1_loss_fn(pred['df'], data['df'])
            df_loss /= self.conf.line_neighborhood
        df_loss = (df_loss * line_mask).sum(dim=[1, 2]) / line_norm
        df_loss *= self.conf.loss_weights.df
        loss += df_loss
        outputs['df_loss'] = df_loss

        # Angle loss, with supervision only on the lines neighborhood
        angle_loss = torch.minimum(
            (pred['line_level'] - data['line_level']) ** 2,
            (np.pi - (pred['line_level'] - data['line_level']).abs()) ** 2)
        angle_loss = (angle_loss * line_mask).sum(dim=[1, 2]) / line_norm
        angle_loss *= self.conf.loss_weights.angle
        loss += angle_loss
        outputs['angle_loss'] = angle_loss

        outputs['total'] = loss
        return outputs

    def metrics(self, pred, data):
        return {}
