import argparse
import copy
import hashlib
import json
import os
import os.path as osp
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .base import show, WireframeGraph


URLS_CKPT = {
    'scalelsd': 'https://huggingface.co/cherubicxn/scalelsd/resolve/main/scalelsd-vitbase-v2-train-sa1b.pt',
    'hawpv3': 'https://github.com/cherubicXN/hawp-torchhub/releases/download/HAWPv3/hawpv3-fdc5487a.pth',
    'deeplsd': 'https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_wireframe.tar',
}

def parse_args():
    p = argparse.ArgumentParser(description='Run multiple line detectors on images')
    # Detector selection and thresholds
    p.add_argument('--detector', '--mode', dest='mode', default='scalelsd',
                  choices=['lsd', 'deeplsd', 'hawpv3', 'scalelsd'],
                  help='Detector to run')
    p.add_argument('-t', '--threshold', default=0.05, type=float,
                  help='Confidence threshold for drawing/selection')

    # IO
    p.add_argument('-i', '--img', required=True, type=str,
                  help='Image file or directory')
    p.add_argument('--pattern', default='*color.jpg', type=str,
                  help='Glob pattern when input is a directory')
    p.add_argument('--width', default=512, type=int)
    p.add_argument('--height', default=512, type=int)
    p.add_argument('--whitebg', default=0.0, type=float)
    p.add_argument('--saveto', default=None, type=str)
    p.add_argument('-e', '--ext', default='pdf', type=str,
                  choices=['pdf', 'png', 'json', 'txt'])
    p.add_argument('--device', default='cuda', type=str,
                  choices=['cuda', 'cpu', 'mps'])
    p.add_argument('--disable-show', default=False, action='store_true')
    p.add_argument('--draw-junctions-only', default=False, action='store_true')
    p.add_argument('--use_lsd', default=False, action='store_true')
    p.add_argument('--use_nms', default=False, action='store_true')

    # Checkpoints and auto-download
    p.add_argument('--ckpt-dir', default=str(Path.home() / '.cache' / 'lipmap' / 'checkpoints'),
                  help='Directory to store/download checkpoints')
    

    return p.parse_args()


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def file_sha256(path: str) -> Optional[str]:
    if not osp.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def try_download(url: Optional[str], dst_path: str) -> bool:
    """Attempt to download a file to dst_path. Returns True on success.
    Requires network connectivity. If url is None, returns False.
    """
    if url is None:
        return False
    try:
        import urllib.request  # lazy import
        ensure_dir(osp.dirname(dst_path))
        print(f'Downloading: {url} -> {dst_path}')
        urllib.request.urlretrieve(url, dst_path)
        return True
    except Exception as e:
        print(f'Warning: download failed for {url}: {e}')
        return False


def prepare_model(args):
    # Resolve device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA requested but not available; falling back to CPU')
        device = torch.device('cpu')
    elif args.device == 'mps':
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    else:
        device = torch.device(args.device)

    # Prepare checkpoint paths
    ensure_dir(args.ckpt_dir)
    scalelsd_ckpt = args.scalelsd_ckpt if osp.isabs(args.scalelsd_ckpt) else osp.join(args.ckpt_dir, args.scalelsd_ckpt)
    deeplsd_ckpt = args.deeplsd_ckpt if osp.isabs(args.deeplsd_ckpt) else osp.join(args.ckpt_dir, args.deeplsd_ckpt)
    hawpv3_ckpt = args.hawpv3_ckpt if osp.isabs(args.hawpv3_ckpt) else osp.join(args.ckpt_dir, args.hawpv3_ckpt)
    hawpv3_cfg = args.hawpv3_cfg

    # Detector-specific setup
    if args.mode == 'scalelsd':
        # Optional auto-download
        if not osp.exists(scalelsd_ckpt):
            ok = try_download(URLS_CKPT['scalelsd'], scalelsd_ckpt)
            if not ok:
                print('Warning: ScaleLSD checkpoint not found and could not be downloaded.')
        # Import and load model
        try:
            # from scalelsd.ssl.models.detector import ScaleLSD  # type: ignore
            from easylsd.models.scalelsd import ScaleLSD  # type: ignore
        except Exception as e:
            raise ImportError('scalelsd package not found. Please install it from https://github.com/ant-research/ScaleLSD') from e
        args.threshold = 10.0
        ScaleLSD.junction_threshold_hm = 0.1
        ScaleLSD.num_junctions_inference = 512
        model = ScaleLSD(gray_scale=True, use_layer_scale=True).eval().to(device)
        if osp.exists(scalelsd_ckpt):
            state_dict = torch.load(scalelsd_ckpt, map_location='cpu')
            try:
                model.load_state_dict(state_dict['model_state'])
            except Exception:
                model.load_state_dict(state_dict)
        else:
            print(f'Note: ScaleLSD checkpoint not found at {scalelsd_ckpt}. Running uninitialized weights.')

    elif args.mode == 'hawpv3':
        if not osp.exists(hawpv3_ckpt):
            ok = try_download(URLS_CKPT['hawpv3'], hawpv3_ckpt)
            if not ok:
                print('Warning: HAWPv3 checkpoint not found and could not be downloaded.')
        # if not osp.exists(hawpv3_cfg):
        #     ok = try_download(args.hawpv3_cfg_url, hawpv3_cfg)
        #     if not ok:
        #         print('Warning: HAWPv3 config not found and could not be downloaded.')
        try:
            # from hawp.fsl.config import cfg as model_config  # type: ignore
            # from hawp.ssl.models import MODELS  # type: ignore
            from easylsd.models import HAWPv3
        except Exception as e:
            raise ImportError('hawp package not found. Please install submodule and package from https://github.com/cherubicXN/hawp') from e
        args.threshold = 0.05
        # if not osp.exists(hawpv3_cfg):
        #     raise FileNotFoundError(f'HAWPv3 config not found: {hawpv3_cfg}. Provide --hawpv3-cfg or enable --download with --hawpv3-cfg-url.')
        # model_config.merge_from_file(hawpv3_cfg)
        
        model = HAWPv3(gray_scale=True).eval().to(device)
        if not osp.exists(hawpv3_ckpt):
            raise FileNotFoundError(f'HAWPv3 checkpoint not found: {hawpv3_ckpt}. Provide --hawpv3-ckpt or enable --download with --hawpv3-ckpt-url.')
        state_dict = torch.load(hawpv3_ckpt, map_location='cpu')
        model.load_state_dict(state_dict)

    elif args.mode == 'lsd':
        model = cv2.createLineSegmentDetector(0)
        # model runs on CPU via OpenCV; keep device for consistency

    elif args.mode == 'deeplsd':
        if not osp.exists(deeplsd_ckpt):
            ok = try_download(URLS_CKPT['deeplsd'], deeplsd_ckpt)
            if not ok:
                print('Warning: DeepLSD checkpoint not found and could not be downloaded.')
        try:
            # from deeplsd.models.deeplsd import DeepLSD  # type: ignore
            from easylsd.models import DeepLSD  # type: ignore
        except Exception as e:
            raise ImportError('deeplsd package not found. Please install from https://github.com/cvg/DeepLSD') from e
        conf = {
            'sharpen': True,
            'detect_lines': True,
            'line_detection_params': {
                'merge': False,
                'optimize': False,
                'use_vps': True,
                'optimize_vps': True,
                'filtering': True,
                'grad_thresh': 3,
                'grad_nfa': True,
            }
        }
        model = DeepLSD(conf).eval().to(device)
        if not osp.exists(deeplsd_ckpt):
            raise FileNotFoundError(f'DeepLSD checkpoint not found: {deeplsd_ckpt}. Provide --deeplsd-ckpt or enable --download with --deeplsd-ckpt-url.')
        state_dict = torch.load(str(deeplsd_ckpt), map_location='cpu')
        model.load_state_dict(state_dict['model'])

    else:
        raise TypeError(f'Unknown detector <{args.mode}>')
    
    return model, device

def main():
    args = parse_args()
    model, device = prepare_model(args)

    # Set up output directory and painter
    if args.saveto is None:
        print('No output directory specified, saving outputs to folder: temp_output/ScaleLSD')
        args.saveto = 'temp_output/ScaleLSD'
    os.makedirs(args.saveto,exist_ok=True)

    if show is None:
        raise ImportError('Visualization utilities not available (scalelsd.base.show or hawp.base.show). Install one of the toolkits to enable drawing/json export.')
    show.painters.HAWPainter.confidence_threshold = args.threshold
    show.painters.HAWPainter.line_width = 3
    show.painters.HAWPainter.marker_size = 4
    show.Canvas.show = not args.disable_show
    if args.whitebg > 0.0:
        show.Canvas.white_overlay = args.whitebg
    painter = show.painters.HAWPainter()
    # edge_color = 'orange' # 'midnightblue'
    # vertex_color = 'Cyan' # 'deeppink'
    edge_color = 'midnightblue' # 'midnightblue'
    vertex_color = 'deeppink' # 'deeppink'

    # Prepare images
    all_images = []
    if osp.isfile(args.img):
        all_images = [args.img]
    elif osp.isdir(args.img):
        all_images = sorted([str(p) for p in Path(args.img).glob(args.pattern)])
    else:
        raise ValueError('Input must be a valid file or a directory.')


    height = args.height
    width = args.width

    for fname in tqdm(all_images):
        pname = Path(fname)
        image = cv2.imread(fname, 0)
        if image is None:
            print(f'Warning: failed to read image {fname}; skipping')
            continue

        ori_shape = image.shape[:2]
        image_resized = cv2.resize(image, (width, height))
        image_t = torch.from_numpy(image_resized).float() / 255.0
        image_t = image_t[None, None].to(device)
        
        meta = {
            'width': ori_shape[1],
            'height':ori_shape[0],
            'filename': '',
            'use_lsd': args.use_lsd,
            'use_nms': args.use_nms,
        }

        with torch.no_grad():
            if args.mode == 'scalelsd':
                outputs, _ = model(image_t, meta)
                outputs = outputs[0]

            elif args.mode == 'hawpv3':
                meta = {
                    'width': ori_shape[1],
                    'height': ori_shape[0],
                    'filename': '',
                    'use_lsd': False,
                }
                outputs, _ = model(image_t, [meta])

            elif args.mode == 'deeplsd':
                img_rgb = cv2.imread(fname)
                if img_rgb is None:
                    print(f'Warning: failed to read image {fname}; skipping')
                    continue
                gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
                inputs = {'image': torch.tensor(gray_img, dtype=torch.float, device=device)[None, None] / 255.}
                out = model(inputs)
                pred_lines = out['lines'][0]
                lines = torch.from_numpy(pred_lines)
                juncs = torch.unique(lines.reshape(-1, 2), dim=0)
                outputs = {
                    'width': ori_shape[1],
                    'height': ori_shape[0],
                    'lines_pred': lines.reshape(-1, 4),
                    'juncs_pred': juncs,
                    'lines_score': torch.ones(lines.shape[0]),
                    'juncs_score': torch.ones(juncs.shape[0]),
                }

            elif args.mode == 'lsd':
                # OpenCV LSD expects uint8 grayscale image
                lsd = model.detect(image)
                if lsd is None or lsd[0] is None:
                    lines = torch.zeros((0, 4), dtype=torch.float32)
                else:
                    lines_arr = lsd[0].reshape(-1, 4)
                    lines = torch.from_numpy(lines_arr)
                juncs = torch.unique(lines.reshape(-1, 2), dim=0) if lines.numel() > 0 else torch.zeros((0, 2), dtype=torch.float32)
                outputs = {
                    'width': ori_shape[1],
                    'height': ori_shape[0],
                    'lines_pred': lines,
                    'juncs_pred': juncs,
                    'lines_score': torch.ones(lines.shape[0]),
                    'juncs_score': torch.ones(juncs.shape[0]),
                }
            else:
                raise TypeError('Please input the correct detector!')

        if args.saveto is not None:
            fig_base = osp.join(args.saveto, pname.stem)
            if args.ext in ['png', 'pdf']:
                fig_file = f'{fig_base}.{args.ext}'
                with show.image_canvas(fname, fig_file=fig_file) as ax:
                    if args.draw_junctions_only:
                        painter.draw_junctions(ax, outputs)
                    else:
                        painter.draw_wireframe(ax, outputs, edge_color=edge_color, vertex_color=vertex_color)
            elif args.ext == 'json':
                if WireframeGraph is None:
                    raise ImportError('WireframeGraph is not available for JSON export.')
                indices = WireframeGraph.xyxy2indices(outputs['juncs_pred'], outputs['lines_pred'])
                wireframe = WireframeGraph(outputs['juncs_pred'], outputs['juncs_score'], indices, outputs['lines_score'], outputs['width'], outputs['height'])
                outpath = f'{fig_base}.json'
                with open(outpath, 'w') as f:
                    json.dump(wireframe.jsonize(), f)
            elif args.ext == 'txt':
                outpath = f'{fig_base}.txt'
                arr = outputs['lines_pred']
                if torch.is_tensor(arr):
                    arr = arr.cpu().numpy()
                np.savetxt(outpath, arr, fmt='%.3f')
            else:
                raise ValueError(f'Unsupported extension: {args.ext} is not in [png, pdf, json, txt]')
        

if __name__ == "__main__":
    main()
