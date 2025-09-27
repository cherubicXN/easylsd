import logging

import numpy as np
import torch 


try:
    import matplotlib
    import matplotlib.animation
    import matplotlib.collections
    import matplotlib.patches
except ImportError:
    matplotlib = None


LOG = logging.getLogger(__name__)


class HAWPainter:
    # line_width = None
    # marker_size = None
    line_width = 2
    marker_size = 4

    confidence_threshold = 0.05

    def __init__(self):

        if self.line_width is None:
            self.line_width = 1
        
        if self.marker_size is None:
            self.marker_size = max(1, int(self.line_width * 0.5))

    def draw_junctions(self, ax, wireframe, *,
            edge_color = None, vertex_color = None):
        if wireframe is None:
            return
        
        if edge_color is None:
            edge_color = 'b'
        if vertex_color is None:
            vertex_color = 'c'
        
        if 'lines_score' in wireframe.keys():
            line_segments = wireframe['lines_pred'][wireframe['lines_score']>self.confidence_threshold]
        else:
            line_segments = wireframe['lines_pred']

        if isinstance(line_segments, torch.Tensor):
            line_segments = line_segments.cpu().numpy()

        ax.plot(line_segments[:,0],line_segments[:,1],'.',color=vertex_color)
        ax.plot(line_segments[:,2],line_segments[:,3],'.',
        color=vertex_color)
    def draw_wireframe(self, ax, wireframe, *,
            edge_color = None, vertex_color = None):
        if wireframe is None:
            return
        
        if edge_color is None:
            edge_color = 'b'
        if vertex_color is None:
            vertex_color = 'c'
        
        if 'lines_score' in wireframe.keys():
            line_segments = wireframe['lines_pred'][wireframe['lines_score']>self.confidence_threshold]
        else:
            line_segments = wireframe['lines_pred']

        # import pdb;pdb.set_trace()    
        if isinstance(line_segments, torch.Tensor):
            line_segments = line_segments.cpu().numpy()

        # import pdb;pdb.set_trace()
        # line_segments = wireframe.line_segments(threshold=self.confidence_threshold)
        # line_segments = line_segments.cpu().numpy()
        ax.plot([line_segments[:,0],line_segments[:,2]],[line_segments[:,1],line_segments[:,3]],'-',color=edge_color,linewidth=self.line_width)
        ax.plot(line_segments[:,0],line_segments[:,1],'.',color=vertex_color,markersize=self.marker_size)
        ax.plot(line_segments[:,2],line_segments[:,3],'.',color=vertex_color,markersize=self.marker_size)
