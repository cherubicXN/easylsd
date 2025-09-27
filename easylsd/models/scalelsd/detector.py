from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from .backbones import build_backbone
from .hafm import HAFMencoder
from .losses import cross_entropy_loss_for_junction, sigmoid_l1_loss

import cv2

class ScaleLSD(nn.Module):
    def __init__(self, gray_scale=False, use_layer_scale=False, enable_attention_hooks=False):

        super(ScaleLSD, self).__init__()

        # Defaults; can be overridden via configure()
        self.num_junctions_inference = 512
        self.junction_threshold_hm = 0.008

        self.distance_threshold = 5.0

        self.hafm_encoder = HAFMencoder(dis_th=self.distance_threshold)

        self.backbone = build_backbone(
            gray_scale=gray_scale,
            use_layer_scale=use_layer_scale,
            enable_attention_hooks=enable_attention_hooks,
        )
        
        self.j2l_threshold = 10

        self.num_residuals = 0

        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.stride = self.backbone.stride
        self.train_step = 0

    @classmethod 
    def configure(cls, opts):
        try:
            cls.num_junctions_inference = getattr(opts, 'num_junctions', cls.num_junctions_inference)
            cls.junction_threshold_hm = getattr(opts, 'junction_hm', cls.junction_threshold_hm)
        except Exception:
            pass

    @classmethod
    def cli(cls, parser):
        try:
            parser.add_argument('-nj', '--num-junctions', default=512, type=int, help='number of junctions')
            parser.add_argument('-jh', '--junction-hm', default=0.008, type=float, help='junction threshold heatmap')
        except Exception:
            pass
    
    def hafm_decoding(self,md_maps, dis_maps, residual_maps, scale=5.0, flatten = True, return_points = False):

        device = md_maps.device
        scale = self.distance_threshold

        batch_size, _, height, width = md_maps.shape
        _y = torch.arange(0,height,device=device).float()
        _x = torch.arange(0,width, device=device).float()

        y0, x0 =torch.meshgrid(_y, _x,indexing='ij')
        y0 = y0[None,None]
        x0 = x0[None,None]
        
        sign_pad = torch.arange(-self.num_residuals,self.num_residuals+1,device=device,dtype=torch.float32).reshape(1,-1,1,1)

        if residual_maps is not None:
            residual = residual_maps*sign_pad
            distance_fields = dis_maps + residual
        else:
            distance_fields = dis_maps
        distance_fields = distance_fields.clamp(min=0,max=1.0)
        md_un = (md_maps[:,:1] - 0.5)*np.pi*2
        st_un = md_maps[:,1:2]*np.pi/2.0
        ed_un = -md_maps[:,2:3]*np.pi/2.0

        cs_md = md_un.cos()
        ss_md = md_un.sin()

        y_st = torch.tan(st_un)
        y_ed = torch.tan(ed_un)

        x_st_rotated = (cs_md - ss_md*y_st)*distance_fields*scale
        y_st_rotated = (ss_md + cs_md*y_st)*distance_fields*scale

        x_ed_rotated = (cs_md - ss_md*y_ed)*distance_fields*scale
        y_ed_rotated = (ss_md + cs_md*y_ed)*distance_fields*scale

        x_st_final = (x_st_rotated + x0).clamp(min=0,max=width-1)
        y_st_final = (y_st_rotated + y0).clamp(min=0,max=height-1)

        x_ed_final = (x_ed_rotated + x0).clamp(min=0,max=width-1)
        y_ed_final = (y_ed_rotated + y0).clamp(min=0,max=height-1)

        
        lines = torch.stack((x_st_final,y_st_final,x_ed_final,y_ed_final),dim=-1)
        if flatten:
            lines = lines.reshape(batch_size,-1,4)
        if return_points:
            points = torch.stack((x0,y0),dim=-1)
            points = points.repeat((batch_size,2*self.num_residuals+1,1,1,1))
            if flatten:
                points = points.reshape(batch_size,-1,2)
            return lines, points
        
        return lines
    
    @staticmethod
    def non_maximum_suppression(a, kernel_size=3):
        ap = F.max_pool2d(a, kernel_size, stride=1, padding=kernel_size//2)
        mask = (a == ap).float().clamp(min=0.0)
        
        return a * mask

    @staticmethod
    def get_junctions(jloc, joff, topk = 300, th=0):
        height, width = jloc.size(1), jloc.size(2)
        jloc = jloc.reshape(-1)
        joff = joff.reshape(2, -1)

        
        scores, index = torch.topk(jloc, k=topk)
        # y = (index // width).float() + torch.gather(joff[1], 0, index) + 0.5
        y = torch.div(index,width,rounding_mode='trunc').float()+ torch.gather(joff[1], 0, index) + 0.5
        x = (index % width).float() + torch.gather(joff[0], 0, index) + 0.5

        junctions = torch.stack((x, y)).t()
        
        if th>0 :
            return junctions[scores>th], scores[scores>th]
        else:
            return junctions, scores

    def wireframe_matcher(self, juncs_pred, lines_pred, hat_points, is_train=False):
        cost1 = torch.sum((lines_pred[:,:2]-juncs_pred[:,None])**2,dim=-1)
        cost2 = torch.sum((lines_pred[:,2:]-juncs_pred[:,None])**2,dim=-1)
        
        dis1, idx_junc_to_end1 = cost1.min(dim=0)
        dis2, idx_junc_to_end2 = cost2.min(dim=0)
        length = torch.sum((lines_pred[:,:2]-lines_pred[:,2:])**2,dim=-1)

        idx_junc_to_end_min = torch.min(idx_junc_to_end1,idx_junc_to_end2)
        idx_junc_to_end_max = torch.max(idx_junc_to_end1,idx_junc_to_end2)

        iskeep = idx_junc_to_end_min < idx_junc_to_end_max ## not the same junction
        if self.j2l_threshold>0:
            iskeep *= (dis1<self.j2l_threshold)*(dis2<self.j2l_threshold)
        
        idx_lines_for_junctions = torch.stack((idx_junc_to_end_min[iskeep],idx_junc_to_end_max[iskeep]),dim=1)#.unique(dim=0)


        global_idx = idx_lines_for_junctions[:,0]*juncs_pred.shape[0]+idx_lines_for_junctions[:,1]
        argsort = torch.argsort(global_idx)
        unique, counts = torch.unique(global_idx[argsort],return_counts=True)
        lines_support = torch.split(lines_pred[iskeep][argsort],counts.tolist())

        hat_points = hat_points[iskeep][argsort]
        hat_points = torch.split(hat_points,counts.tolist())


        # ux = unique//juncs_pred.shape[0]
        ux = torch.div(unique, juncs_pred.shape[0], rounding_mode='trunc')
        uy = unique%juncs_pred.shape[0]
        uxy = torch.stack((ux,uy),dim=1)
        
        lines_adjusted = juncs_pred[uxy].reshape(-1,4)
        return lines_adjusted, uxy, lines_support, hat_points, counts


    def forward_backbone(self, images):
        outputs, features = self.backbone(images)
        if isinstance(outputs, list):
            auxputs = outputs[1:]
            outputs = outputs[0]
        else:
            auxputs = []

        return outputs, features, auxputs

    @torch.no_grad()
    def detect_junctions(self, images, junction_heatmaps = None):
        device = images.device
        output, features, aux = self.forward_backbone(images)
        joff_pred = output[:,7:9].sigmoid()-0.5
        if junction_heatmaps is None:
            jloc_pred = output[:,5:7].softmax(1)[:,1:]
        else:
            jloc_pred = junction_heatmaps
        
        batch_size = images.shape[0]

        junctions_batch = []
        for i in range(batch_size):
            jloc_pred_nms = self.non_maximum_suppression(jloc_pred[i])
            junctions, scores = self.get_junctions(jloc_pred_nms,joff_pred[i], topk=self.num_junctions_inference,th=self.junction_threshold_hm)
            junctions_batch.append(junctions)
        
        return junctions_batch
        
    @torch.no_grad()
    def compute_hatlines(self, images):
        device = images.device
        output, features, aux = self.forward_backbone(images)
        md_pred = output[:,:3].sigmoid()
        dis_pred = output[:,3:4].sigmoid()
        res_pred = output[:,4:5].sigmoid()
        lines_pred_batch, hat_points_batch = self.hafm_decoding(md_pred, dis_pred, None, flatten = True, return_points=True)
        return lines_pred_batch, hat_points_batch

    def forward(self, images, annotations = None, targets = None):
        if self.training:
            return self.forward_train(images, annotations=annotations)
        else:
            return self.forward_test(images, annotations=annotations)

    def compute_loss(self, output, targets, mask, loss_dict):
        # for nstack, output in enumerate(outputs):
        
        loss_map = torch.mean(F.l1_loss(output[:,:3].sigmoid(), targets['md'],reduction='none'),dim=1,keepdim=True)
        loss_dict['loss_md']  += torch.mean(loss_map*mask) / (torch.mean(mask)+1e-6)
        loss_map = F.l1_loss(output[:,3:4].sigmoid(), targets['dis'], reduction='none')
        loss_dict['loss_dis'] += torch.mean(loss_map*mask) / (torch.mean(mask)+1e-6)
        loss_residual_map = F.l1_loss(output[:,4:5].sigmoid(), loss_map, reduction='none')
        loss_dict['loss_res'] += torch.mean(loss_residual_map*mask)/(torch.mean(mask)+1e-6)
        loss_dict['loss_jloc'] += cross_entropy_loss_for_junction(output[:,5:7], targets['jloc'])
        loss_dict['loss_joff'] += sigmoid_l1_loss(output[:,7:9], targets['joff'], -0.5, targets['jloc'])

        return loss_dict
    

    def forward_train(self, images, annotations = None):
        batch_size = images.size(0)
        self.train_step += 1

        valid_mask = annotations['valid_mask']
                
        targets , metas = self.hafm_encoder(annotations)

        outputs, features, auxputs = self.forward_backbone(images)

        loss_dict = {
            'loss_md': 0.0,
            'loss_dis': 0.0,
            'loss_res': 0.0,
            'loss_jloc': 0.0,
            'loss_joff': 0.0,
        }

        extra_info = defaultdict(list)

        mask = targets['mask']

        loss_dict = self.compute_loss(outputs, targets, mask, loss_dict)
        if len(auxputs)>0:
            for auxput in auxputs:
                loss_dict = self.compute_loss(auxput, targets, mask, loss_dict)
        
        for key in extra_info.keys():
            extra_info[key] = extra_info[key]/batch_size

        return loss_dict, extra_info


    @torch.no_grad()
    def forward_test(self, images, annotations=None, merge=False):
        device = images.device
        batch_size, _, height, width = images.shape

        outputs, features, aux = self.forward_backbone(images)

        if annotations is None:
            annotations = {}
        annotations.setdefault("use_lsd", True)
        annotations.setdefault("use_nms", True)

        # use lsd for theta prediction
        if annotations['use_lsd']:
            ws = images.shape[3]//self.stride 
            hs = images.shape[2]//self.stride 
            lsd = cv2.createLineSegmentDetector(0)        
            
            md_lsd_batch = []
            dis_lsd_batch = []
            for i in range(batch_size):
                image = np.array(images[i,0].cpu().numpy()*255,dtype=np.uint8)
                det = lsd.detect(image)
                lsd_lines = det[0].reshape(-1,4) if det is not None and det[0] is not None else np.zeros((0,4), dtype=np.float32)
                
                # transform lsd lines to lsd-hat-field 
                md_lsd, dis_lsd, _ = self.hafm_encoder.lines2hafm(torch.from_numpy(lsd_lines).to(images.device)/self.stride, hs, ws)
                md_lsd_batch.append(md_lsd)
                dis_lsd_batch.append(dis_lsd)

            md_pred = torch.stack(md_lsd_batch, dim=0)
            dis_pred = torch.stack(dis_lsd_batch, dim=0)

            # for junctions/endpoints extraction
            md_pred[:,1:3] = outputs[:,1:3].sigmoid()
            # dist
            dis_pred = outputs[:,3:4].sigmoid()

            jloc_pred= outputs[:,5:7].softmax(1)[:,1:]
            joff_pred= outputs[:,7:9].sigmoid() - 0.5
        else:
     
            md_pred = outputs[:,:3].sigmoid()
            dis_pred = outputs[:,3:4].sigmoid()
            jloc_pred= outputs[:,5:7].softmax(1)[:,1:]
            joff_pred= outputs[:,7:9].sigmoid() - 0.5

        lines_pred_batch, hat_points_batch = self.hafm_decoding(md_pred, dis_pred, None, flatten = True, return_points=True)

        output_list = []
        graph_pred = torch.zeros((batch_size, self.num_junctions_inference, self.num_junctions_inference), device=images.device)
        for i in range(batch_size):
            if annotations['use_nms']:
                jloc_pred_nms = self.non_maximum_suppression(jloc_pred[i])
            else:
                jloc_pred_nms = self.non_maximum_suppression(jloc_pred[i], kernel_size=1)
            topK = min(self.num_junctions_inference, int((jloc_pred_nms>self.junction_threshold_hm).float().sum().item()))
            juncs_pred, juncs_score = self.get_junctions(jloc_pred_nms,joff_pred[i], topk=topK, th=self.junction_threshold_hm)

            lines_adjusted, indices_adj, supports, hat_points, counts = self.wireframe_matcher(juncs_pred, lines_pred_batch[i], hat_points_batch[i])

            jscales = torch.tensor(
                [
                    annotations['width']/md_pred.size(3),
                    annotations['height']/md_pred.size(2)
                ],
                device=images.device
            )

            junctions = juncs_pred * jscales
            supports = [_*self.stride for _ in supports]

            num_junctions = junctions.shape[0]
            graph_pred[i, indices_adj[:,0], indices_adj[:,1]] += counts.float()
            graph_pred[i, indices_adj[:,1], indices_adj[:,0]] += counts.float()
            graph_i = graph_pred[i,:num_junctions,:num_junctions]
            edges = graph_i.triu().nonzero()
            lines = junctions[edges].reshape(-1,4)
            scores = graph_pred[i, edges[:,0], edges[:,1]]

            output_list.append(
                {
                    'lines_pred': lines,
                    'lines_score': scores,
                    'juncs_pred': junctions,
                    'lines_support': supports,
                    'juncs_score': juncs_score,
                    'graph': graph_i,
                    'width': annotations['width'],
                    'height': annotations['height'],
                }
            )
        
        return output_list, {}
    
