from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import time

from .backbones import build_backbone_easy

from .base import HAWPBase
from .heatmap_decoder import PixelShuffleDecoder


class HAWPv3(HAWPBase):
    def __init__(self, gray_scale=True):
        super(HAWPv3, self).__init__(
            num_points = 32,
            num_residuals = 1,
            distance_threshold =  2.0)

        self.backbone = build_backbone_easy(gray_scale=gray_scale)

        #Matcher
        self.j2l_threshold = 10.0
        self.jmatch_threshold = 1.5

        # LOI POOLING
        self.use_init_lines = True 
        self.dim_junction_feature    = 128
        self.dim_edge_feature = 4
        self.dim_fc     = 1024


        self.n_out_junc = 250
        self.n_out_line = 2500

        self.num_junctions_inference = 300
        self.junction_threshold_hm = 0.008
        self.use_residual = 1
        
        self.loi_cls_type = 'softmax'
        self.loi_layer_norm = False 
        self.loi_activation = nn.ReLU 

        self.fc1 = nn.Conv2d(256, self.dim_junction_feature, 1)

        self.fc3 = nn.Conv2d(256, self.dim_edge_feature, 1)
        self.fc4 = nn.Conv2d(256, self.dim_edge_feature, 1)

        self.regional_head = nn.Conv2d(256, 1, 1)
        fc2 = [nn.Linear(self.dim_junction_feature*2 + (self.num_points-2)*self.dim_edge_feature*(1+self.use_init_lines), self.dim_fc),
        ]
        for i in range(2):
            fc2.append(nn.ReLU(True))
            fc2.append(nn.Linear(self.dim_fc,self.dim_fc))

        
        self.fc2 = nn.Sequential(*fc2)
        self.fc2_res = nn.Sequential(nn.Linear((1+self.use_init_lines)*(self.num_points-2)*self.dim_edge_feature, self.dim_fc),nn.ReLU(True))

        self.line_mlp = nn.Sequential(
            nn.Linear((self.num_points-2)*self.dim_edge_feature,128),
            nn.ReLU(True),
            nn.Linear(128,32),nn.ReLU(True),
            nn.Linear(32,1)
        )

        if self.loi_cls_type == 'softmax':
            self.fc2_head = nn.Linear(self.dim_fc, 2)
            self.loss = nn.CrossEntropyLoss(reduction='none')
        elif self.loi_cls_type == 'sigmoid':
            self.fc2_head = nn.Linear(self.dim_fc, 1)
            self.loss = nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise NotImplementedError()
        
        self.use_heatmap_decoder = True
        if self.use_heatmap_decoder:
            self.heatmap_decoder = PixelShuffleDecoder(input_feat_dim=256)

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.train_step = 0

    
    def wireframe_matcher(self, juncs_pred, lines_pred, is_train=False, is_shuffle=False):
        cost1 = torch.sum((lines_pred[:,:2]-juncs_pred[:,None])**2,dim=-1)
        cost2 = torch.sum((lines_pred[:,2:]-juncs_pred[:,None])**2,dim=-1)
        
        dis1, idx_junc_to_end1 = cost1.min(dim=0)
        dis2, idx_junc_to_end2 = cost2.min(dim=0)
        length = torch.sum((lines_pred[:,:2]-lines_pred[:,2:])**2,dim=-1)


        idx_junc_to_end_min = torch.min(idx_junc_to_end1,idx_junc_to_end2)
        idx_junc_to_end_max = torch.max(idx_junc_to_end1,idx_junc_to_end2)

        iskeep = idx_junc_to_end_min < idx_junc_to_end_max
        if self.j2l_threshold>0:
            iskeep *= (dis1<self.j2l_threshold)*(dis2<self.j2l_threshold)

        idx_lines_for_junctions = torch.stack((idx_junc_to_end_min[iskeep],idx_junc_to_end_max[iskeep]),dim=1)#.unique(dim=0)


        if is_shuffle:
            cost_atoi_argsort = torch.randperm(iskeep.sum(),device=juncs_pred.device)
        else:
            cost_atoi = torch.min(
             torch.sum((juncs_pred[idx_lines_for_junctions].reshape(-1,4) - lines_pred[iskeep])**2,dim=-1),
             torch.sum((juncs_pred[idx_lines_for_junctions].reshape(-1,4) - lines_pred[iskeep][:,[2,3,0,1]])**2,dim=-1)
             )

            cost_atoi_argsort = cost_atoi.argsort(descending=True)
            

        lines_pred_kept = lines_pred[iskeep][cost_atoi_argsort]
        idx_lines_for_junctions = idx_lines_for_junctions[cost_atoi_argsort]

        _, perm = np.unique(idx_lines_for_junctions.cpu().numpy(),return_index=True,axis=0)

        idx_lines_for_junctions = idx_lines_for_junctions[perm]
        lines_init = lines_pred_kept[perm]
        
        if is_train:
            idx_lines_for_junctions_mirror = torch.cat((idx_lines_for_junctions[:,1,None],idx_lines_for_junctions[:,0,None]),dim=1)
            idx_lines_for_junctions = torch.cat((idx_lines_for_junctions, idx_lines_for_junctions_mirror))
        
        #lines_adjusted = torch.cat((juncs_pred[idx_lines_for_junctions[:,0]], juncs_pred[idx_lines_for_junctions[:,1]]),dim=1)
        lines_adjusted = juncs_pred[idx_lines_for_junctions].reshape(-1,4)
        #lines_adjusted = torch.stack((juncs_pred[idx_lines_for_junctions[:,0]], juncs_pred[idx_lines_for_junctions[:,1]]),dim=1)
        
        return lines_adjusted, lines_init, perm, idx_lines_for_junctions

    @torch.no_grad()
    def forward_test_with_junction(self, images, junctions, annotations = None):
        device = images.device

        extra_info = {
            'time_backbone': 0.0,
            'time_proposal': 0.0,
            'time_matching': 0.0,
            'time_verification': 0.0,
        }

        extra_info['time_backbone'] = time.time()

        outputs, features = self.backbone(images)

        loi_features = self.fc1(features)
        loi_features_thin = self.fc3(features)
        loi_features_aux = self.fc4(features)

        
        output = outputs[0]
        md_pred = output[:,:3].sigmoid()
        dis_pred = output[:,3:4].sigmoid()
        res_pred = output[:,4:5].sigmoid()
        jloc_pred= output[:,5:7].softmax(1)[:,1:]
        # jloc_logits = output[:,5:7].softmax(1)
        joff_pred= output[:,7:9].sigmoid() - 0.5

        extra_info['time_backbone'] = time.time() - extra_info['time_backbone']

        batch_size = md_pred.size(0)
        
        lines_pred = self.hafm_decoding(md_pred, dis_pred, res_pred if self.use_residual else None, flatten = True)[0]

        
        juncs_pred = junctions
        
        _ = torch.ones((juncs_pred.shape[0]),dtype=juncs_pred.dtype,device=device)
        

        
        lines_adjusted, lines_init, perm, _ = self.wireframe_matcher(juncs_pred, lines_pred)
        
        e1_features = self.bilinear_sampling(loi_features[0], lines_adjusted[:,:2]-0.5).t()
        e2_features = self.bilinear_sampling(loi_features[0], lines_adjusted[:,2:]-0.5).t()

        f1 = self.compute_loi_features(loi_features_thin[0],lines_adjusted, tspan=self.tspan[...,1:-1])
        line_features = torch.cat((e1_features,e2_features,f1),dim=-1)

        if self.use_init_lines:
            f2 = self.compute_loi_features(loi_features_aux[0],lines_init, tspan=self.tspan[...,1:-1])
            line_features = torch.cat((line_features,f2),dim=-1)
            logits = self.fc2_head(self.fc2(line_features)+self.fc2_res(torch.cat((f1,f2),dim=-1)))
        else:
            logits = self.fc2_head(self.fc2(line_features)+self.fc2_res(f1))

        if self.loi_cls_type == 'softmax':
            scores = logits.softmax(dim=-1)[:,1]
        else:
            scores = logits.sigmoid()[:,0]
        
        sarg = torch.argsort(scores,descending=True)

        lines_final = lines_adjusted[sarg]
        score_final = scores[sarg]
        lines_before = lines_init[sarg]

        num_detection = min((score_final>0.00).sum(),1000)
        lines_final = lines_final[:num_detection]
        score_final = score_final[:num_detection]

        juncs_final = juncs_pred
        juncs_score = _

        # Determine output scaling
        if isinstance(annotations, list) and len(annotations) > 0:
            ann0 = annotations[0]
        elif isinstance(annotations, dict):
            ann0 = annotations
        else:
            ann0 = {}
        sx = ann0.get('width', images.size(3)) / output.size(3)
        sy = ann0.get('height', images.size(2)) / output.size(2)

        lines_final[:,0] *= sx
        lines_final[:,1] *= sy
        lines_final[:,2] *= sx
        lines_final[:,3] *= sy

        juncs_final[:,0] *= sx
        juncs_final[:,1] *= sy

        output = {
            'lines_pred': lines_final,
            'lines_score': score_final,
            'juncs_pred': juncs_final,
            'juncs_score': juncs_score,
            'num_proposals': lines_adjusted.size(0),
            'filename': ann0.get('filename'),
            'width': ann0.get('width', images.size(3)),
            'height': ann0.get('height', images.size(2)),
        }

        return output, extra_info
    @torch.no_grad()
    def forward_test(self, images, annotations = None):
        device = images.device

        extra_info = {
            'time_backbone': 0.0,
            'time_proposal': 0.0,
            'time_matching': 0.0,
            'time_verification': 0.0,
        }

        extra_info['time_backbone'] = time.time()

        outputs, features = self.backbone(images)

        loi_features = self.fc1(features)
        loi_features_thin = self.fc3(features)
        loi_features_aux = self.fc4(features)

        output = outputs[0]
        md_pred = output[:,:3].sigmoid()
        dis_pred = output[:,3:4].sigmoid()
        res_pred = output[:,4:5].sigmoid()
        jloc_pred= output[:,5:7].softmax(1)[:,1:]
        # jloc_logits = output[:,5:7].softmax(1)
        joff_pred= output[:,7:9].sigmoid() - 0.5

        extra_info['time_backbone'] = time.time() - extra_info['time_backbone']

        batch_size = md_pred.size(0)
        
        lines_pred = self.hafm_decoding(md_pred, dis_pred, res_pred if self.use_residual else None, flatten = True)[0]


        jloc_pred_nms = self.non_maximum_suppression(jloc_pred[0])

        #building
        # self.num_junctions_inference = 512
        topK = min(self.num_junctions_inference, int((jloc_pred_nms>self.junction_threshold_hm).float().sum().item()))
        
        juncs_pred, _ = self.get_junctions(jloc_pred_nms,joff_pred[0], topk=topK,th=self.junction_threshold_hm)
        if juncs_pred.shape[0] == 0:
            return {'lines_pred': None, 'lines_score': None}
        lines_adjusted, lines_init, perm, _ = self.wireframe_matcher(juncs_pred, lines_pred)
        matcher_time = time.time()
        # lines_adjusted_trials = []
        # lines_init_trials = []
        # for i in range(10):
        #     lines_adjusted, lines_init, perm = self.wireframe_matcher(juncs_pred, lines_pred)
        #     lines_adjusted_trials.append(lines_adjusted)
        #     lines_init_trials.append(lines_init)
        # matcher_time = time.time()-matcher_time
        # lines_init_trials = torch.stack(lines_init_trials)
        # is_stable = lines_init_trials.std(dim=0).mean(dim=-1)<1e-6
        
        e1_features = self.bilinear_sampling(loi_features[0], lines_adjusted[:,:2]-0.5).t()
        e2_features = self.bilinear_sampling(loi_features[0], lines_adjusted[:,2:]-0.5).t()

        f1 = self.compute_loi_features(loi_features_thin[0],lines_adjusted, tspan=self.tspan[...,1:-1])
        if self.use_init_lines:
            f2 = self.compute_loi_features(loi_features_aux[0],lines_init, tspan=self.tspan[...,1:-1])
            line_features = torch.cat((e1_features,e2_features,f1,f2),dim=-1)
            logits = self.fc2_head(self.fc2(line_features)+self.fc2_res(torch.cat((f1,f2),dim=-1)))
        else:
            line_features = torch.cat((e1_features,e2_features,f1),dim=-1)
            logits = self.fc2_head(self.fc2(line_features)+self.fc2_res(f1))


        if self.loi_cls_type == 'softmax':
            scores = logits.softmax(dim=-1)[:,1]
        else:
            scores = logits.sigmoid()[:,0]

        sarg = torch.argsort(scores,descending=True)

        lines_final = lines_adjusted[sarg]
        score_final = scores[sarg]
        lines_before = lines_init[sarg]

        num_detection = min((score_final>0.00).sum(),1000)
        lines_final = lines_final[:num_detection]
        score_final = score_final[:num_detection]

        juncs_final = juncs_pred
        juncs_score = _

        if isinstance(annotations, list) and len(annotations) > 0:
            ann0 = annotations[0]
        elif isinstance(annotations, dict):
            ann0 = annotations
        else:
            ann0 = {}
        sx = ann0.get('width', images.size(3)) / output.size(3)
        sy = ann0.get('height', images.size(2)) / output.size(2)

        lines_final[:,0] *= sx
        lines_final[:,1] *= sy
        lines_final[:,2] *= sx
        lines_final[:,3] *= sy

        juncs_final[:,0] *= sx
        juncs_final[:,1] *= sy

        output = {
            'lines_pred': lines_final,
            'lines_score': score_final,
            'juncs_pred': juncs_final,
            'juncs_score': juncs_score,
            'num_proposals': lines_adjusted.size(0),
            'filename': ann0.get('filename'),
            'width': ann0.get('width', images.size(3)),
            'height': ann0.get('height', images.size(2)),
        }
        return output, extra_info

    def forward(self, images, annotations = None, targets = None):
        return self.forward_test(images, annotations=annotations)