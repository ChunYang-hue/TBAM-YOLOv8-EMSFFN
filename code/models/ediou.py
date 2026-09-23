import torch


def ediou_loss(pred, target, xywh=True, eps=1e-7):
    """EDIoU loss based on endpoint distance.

    pred, target: (N, 4)
    If xywh=True: [x_center, y_center, w, h]
    If xywh=False: [x1, y1, x2, y2]

    L_EDIoU = 1 - IoU + (c1^2 + c2^2) / (w^2 + h^2)
    c1^2 = (x1_gt - x1)^2 + (y1_gt - y1)^2
    c2^2 = (x2_gt - x2)^2 + (y2_gt - y2)^2
    """
    if xywh:
        px, py, pw, ph = pred.unbind(-1)
        tx, ty, tw, th = target.unbind(-1)

        px1 = px - pw / 2
        py1 = py - ph / 2
        px2 = px + pw / 2
        py2 = py + ph / 2

        tx1 = tx - tw / 2
        ty1 = ty - th / 2
        tx2 = tx + tw / 2
        ty2 = ty + th / 2
    else:
        px1, py1, px2, py2 = pred.unbind(-1)
        tx1, ty1, tx2, ty2 = target.unbind(-1)
        pw = px2 - px1
        ph = py2 - py1

    # IoU
    inter_x1 = torch.max(px1, tx1)
    inter_y1 = torch.max(py1, ty1)
    inter_x2 = torch.min(px2, tx2)
    inter_y2 = torch.min(py2, ty2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = area_p + area_t - inter + eps
    iou = inter / union

    # Endpoint distances
    c1 = (px1 - tx1) ** 2 + (py1 - ty1) ** 2
    c2 = (px2 - tx2) ** 2 + (py2 - ty2) ** 2

    # Normalization term
    denom = (pw ** 2 + ph ** 2).clamp(min=eps)

    loss = 1.0 - iou + (c1 + c2) / denom
    return loss.mean()
