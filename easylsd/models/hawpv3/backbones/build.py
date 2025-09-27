from .stacked_hg import HourglassNet, Bottleneck2D
from .multi_task_head import MultitaskHead

def build_backbone_easy(gray_scale):
    inplanes = 64
    num_feats = 128
    depth = 4
    num_stacks = 2
    num_blocks = 1
    head_size = [[3], [1], [1], [2], [2]]


    out_feature_channels = 256

    if gray_scale:
        input_channels = 1
    else:
        input_channels = 3
    num_class = sum(sum(head_size, []))
    model = HourglassNet(
        input_channels=input_channels,
        block=Bottleneck2D,
        inplanes = inplanes,
        num_feats= num_feats,
        depth=depth,
        head=lambda c_in, c_out: MultitaskHead(c_in, c_out, head_size=head_size),
        num_stacks = num_stacks,
        num_blocks = num_blocks,
        num_classes = num_class)

    model.out_feature_channels = out_feature_channels

    return model