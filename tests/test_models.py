import pytest
import torch

from neurodx.models import ARCHS, BACKBONES, build_model


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("arch", ARCHS)
def test_output_shapes(arch, backbone):
    model = build_model(arch, backbone, num_cls_classes=4, num_seg_classes=2, pretrained=False).eval()
    x = torch.randn(2, 3, 96, 96)
    with torch.no_grad():
        seg, cls = model(x)
    assert seg.shape == (2, 2, 96, 96)
    assert cls.shape == (2, 4)


def test_unknown_options_raise():
    with pytest.raises(ValueError):
        build_model("multitask", "resnet9000", 2, pretrained=False)
    with pytest.raises(ValueError):
        build_model("transformer", "vgg16", 2, pretrained=False)
