import torch


def line_iou(pred, target, img_w, length=15, aligned=True):
    '''
    Calculate the line iou value between predictions and targets
    Args:
        pred: lane predictions, shape: (num_pred, 72)
        target: ground truth, shape: (num_target, 72)
        img_w: image width
        length: extended radius
        aligned: True for iou loss calculation, False for pair-wise ious in assign
    '''
    px1 = pred - length
    px2 = pred + length
    tx1 = target - length
    tx2 = target + length
    if aligned:
        invalid_mask = target
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)
    else:
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)
        ovr = (torch.min(px2[:, None, :], tx2[None, ...]) -
               torch.max(px1[:, None, :], tx1[None, ...]))
        union = (torch.max(px2[:, None, :], tx2[None, ...]) -
                 torch.min(px1[:, None, :], tx1[None, ...]))

    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)
    ovr[invalid_masks] = 0.
    union[invalid_masks] = 0.
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
    return iou


def compute_angel(pred, target, img_h):
    # pred -> [num_pred,72]
    # target -> [num_target,72]
    pred_y = torch.zeros_like(pred).cuda()
    pred_y_ = torch.linspace(0, img_h, 72).cuda()
    pred_y[:] = pred_y_.cuda()

    target_y = torch.zeros_like(target).cuda()
    target_y_ = torch.linspace(0, img_h, 72).cuda()
    target_y[:] = target_y_.cuda()

    pred_grad = (pred_y[:, 1:] - pred_y[:, :-1]) / ((pred[:, 1:] - pred[:, :-1]) + 2e-9).cuda()
    # pred_grad = torch.cat((torch.zeros(pred.shape[0],1).cuda(),pred_grad),dim = -1).cuda()

    target_grad = (target_y[:, 1:] - target_y[:, :-1]) / ((target[:, 1:] - target[:, :-1]) + 2e-9).cuda()
    # target_grad = torch.cat((torch.zeros(target.shape[0], 1).cuda(), target_grad), dim=-1).cuda()

    line_angel = torch.abs(pred_grad - target_grad) / (1 + pred_grad * target_grad).cuda()
    line_angel = (1 - torch.cos(torch.atan(line_angel))).cuda()

    angel_loss = line_angel.mean(dim=-1).cuda()
    return angel_loss


def wliou_loss(pred, target, img_w, img_h, length=15):
    w = compute_angel(pred, target, img_h)
    w = (w / torch.sum(w)) + 1e-9

    loss = 1 - line_iou(pred, target, img_w, length)

    loss = loss * w

    return loss.mean()
