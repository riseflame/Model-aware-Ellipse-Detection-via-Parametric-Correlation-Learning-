import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torch.autograd import Variable
import os
import math
def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4) # 4
    loss = 0
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, pred_tensor, target_tensor):
        return self.neg_loss(pred_tensor, target_tensor)


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, pred, mask, ind, target):
        pred = _transpose_and_gather_feat(pred, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.smooth_l1_loss(pred * mask, target * mask, reduction='sum')
        return loss / (mask.sum() + 1e-8)
    
class RegL1Loscs_offset(nn.Module):
    def __init__(self):
        super(RegL1Loscs_offset, self).__init__()

    def forward(self, pred, mask, ind, target,beta = 1.0/9.0):
        pred = _transpose_and_gather_feat(pred, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.smooth_l1_loss(pred * mask, target * mask, reduction='sum',beta = beta)
        return loss / (mask.sum() + 1e-8)


class RegL1Loss_ang(nn.Module):
    def __init__(self):
        super(RegL1Loss_ang, self).__init__()

    def forward(self, pred, mask, ind, target, pred_ab):
        pred_ang = _transpose_and_gather_feat(pred, ind)
        mask_ang = mask.unsqueeze(2).expand_as(pred_ang).float()
        loss = F.smooth_l1_loss(pred_ang * mask_ang, target * mask_ang, reduction='none')

        pred_ab = _transpose_and_gather_feat(pred_ab, ind)
        mask_ab = mask.unsqueeze(2).expand_as(pred_ab).float()
        F.relu(pred_ab, inplace=True)

        ab_ratio = ((pred_ab * mask_ab)[:, :, 0] / ((pred_ab * mask_ab)[:, :, 1] + 1e-8)).reshape((-1, 100, 1))
        ab_ratio.clamp_(min=1, max=10)
        ab_ratio = torch.where(ab_ratio < 1.2, 1, 2)

        loss = torch.sum(loss * ab_ratio)
        return loss / (mask.sum() + 1e-8)


def trace(A):
    return A.diagonal(dim1=-2, dim2=-1).sum(-1)


def sqrt_newton_schulz_autograd(A, numIters, dtype):
    batchSize = A.data.shape[0]
    dim = A.data.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A)).cuda()
    I = Variable(torch.eye(dim, dim).view(1, dim, dim).
                 repeat(batchSize, 1, 1).type(dtype), requires_grad=False).cuda()
    Z = Variable(torch.eye(dim, dim).view(1, dim, dim).
                 repeat(batchSize, 1, 1).type(dtype), requires_grad=False).cuda()

    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA


class Compactness_loss(nn.Module):
    def __init__(self):
        super(Compactness_loss, self).__init__()
        self.loss = nn.SmoothL1Loss()
        self.sig = nn.Sigmoid()
    def forward(self,pred,param,target_mask ):
        j = -1
        loss = 0
        #图像
        for instance in param:
            j += 1
            #目标
            for i in range(int(instance[0])):
                index = i*5+1
                mask = np.zeros((1, 128, 128), dtype=np.float32)
                cv2.ellipse(mask[0], (int(instance[index]), int(instance[index+1])), (int(instance[index+2]), int(instance[index+3])), int(instance[index+4]),
                            0, 360, 1, -1)
                mask_index = np.nonzero(mask[0])
                #x_max = mask_index[0].max()
                #y_max = mask_index[1].max()
                #x_min = mask_index[0].min()
                #y_min = mask_index[1].min()
                #slice_pred = torch.unsqueeze(pred[j,:,x_min:x_max+1,y_min:y_max+1],-1)
                #slice_target = torch.unsqueeze(target_mask[j,:,x_min:x_max+1,y_min:y_max+1],-1)
                #para = [instance[index],instance[index+1],instance[index+2],instance[index+3],instance[index+4]]
                #loss += self.get_compactness_cost_v2(torch.unsqueeze(pred[j,:,:,:],-1),nonzero_indices, target_mask[j],slice_pred)[0]
                #loss += self.get_compactness_cost_v3(slice_pred,slice_target)
                edge = np.zeros((1, 128, 128), dtype=np.float32)
                cv2.ellipse(edge[0], (int(instance[index]), int(instance[index+1])), (int(instance[index+2]), int(instance[index+3])), int(instance[index+4]),
                            0, 360, 1, 1)
                edge_index = np.nonzero(edge)
                loss += self.get_compactness_cost_v4(torch.unsqueeze(pred[j,:,:,:],-1),target_mask[j],edge_index,mask_index)

            
        return loss
    def get_compactness_cost_v4(self,pred,target,edge_index,mask_index): 

        x_indexes_dict = {}
        y_indexes_dict = {}
        epsilon = 1e-8  # parameter to avoid square root of zero
        gt_p = torch.sum(target[:,edge_index[1],edge_index[2]])
        gt_s = torch.sum(target[:,mask_index[0],mask_index[1]])
        gt_r = gt_p / gt_s
        #pre_x = slice_pred[:, 1:, :] - slice_pred[:, :-1, :]  # horizontal direction
        #pre_y = slice_pred[:, :, 1:] - slice_pred[:, :, :-1]  # vertical direction

        pr_p = torch.sum(pred[:,edge_index[1],edge_index[2]])
        pr_s = torch.sum(pred[:,mask_index[0],mask_index[1]])
        pr_r = pr_p / pr_s + epsilon

        compactness_loss = 1 - 1 / (1 + torch.log(1+(pr_r-gt_r)**2 ))

        return compactness_loss
    
    
    def get_compactness_cost_v3(self,slice_pred,slice_target):  
        


        pre_x = slice_pred[:, 1:, :] - slice_pred[:, :-1, :]  # horizontal direction
        pre_y = slice_pred[:, :, 1:] - slice_pred[:, :, :-1]  # vertical direction

        delta_x = pre_x[:, :, 1:] ** 2
        delta_y = pre_y[:, 1:, :] ** 2
        delta_u = torch.abs(delta_x + delta_y)
        epsilon = 1e-8  # parameter to avoid square root of zero
        w = 0.1

        pre_length = w * torch.sum(torch.sqrt(delta_u + epsilon), dim=[1, 2])
        pre_area = torch.sum(slice_pred, dim=[1, 2])
        pre_comp = torch.sum(pre_length ** 2 / (pre_area * 4 * 3.1415926))

        tar_x = slice_target[:, 1:, :] - slice_target[:, :-1, :]  # horizontal direction
        tar_y = slice_target[:, :, 1:] - slice_target[:, :, :-1]  # vertical direction
        delta_x = tar_x[:, :, 1:] ** 2
        delta_y = tar_y[:, 1:, :] ** 2
        delta_u = torch.abs(delta_x + delta_y)

        tar_length = w * torch.sum(torch.sqrt(delta_u + epsilon), dim=[1, 2])
        tar_area = torch.sum(slice_target, dim=[1, 2])
        tar_comp = torch.sum(tar_length ** 2 / (tar_area * 4 * 3.1415926))



        #compactness_loss = torch.sum(length ** 2 / (area * 4 * 3.1415926))
        #compactness_loss = self.loss(length/(area+ epsilon),ratio) 
        #compactness_loss = torch.abs(length/(area+ epsilon) - ratio)
        #compactness_loss = 1 - 1 / (1 + torch.log((pre_comp/(tar_comp+ epsilon))))
        compactness_loss = self.loss(pre_comp , tar_comp)
        #compactness_loss = math.e**(-torch.log(length/(area+ epsilon)) / torch.log(ratio) + epsilon)
        #compactness_loss = self.sig(compactness_loss)
        return compactness_loss
    

    def get_compactness_cost(self,y_pred,nonzero_indices):
        # Assuming y_pred is a PyTorch tensor of shape BxHxWxC
        y_pred = y_pred[..., 0]
        
        # nonzero_indices = torch.tensor(nonzero_indices).to("cuda")
        x_index = set(nonzero_indices[0])
        y_index = set(nonzero_indices[1])
        y = 0
        x = 0
        size = len(nonzero_indices[0])
        for item in x_index:
            index = []
            indexes = [index for index, element in enumerate(nonzero_indices[0]) if element == item]
            index.append(indexes)
            index = index[0]
            if(len(index) > 1):
                #此index为nonzero_indices中的索引
                y += torch.sum(y_pred[:, int(item),nonzero_indices[1][index[1]:index[-1]+1]] - y_pred[:, int(item), nonzero_indices[1][index[0]:index[-1]]])
        for item in y_index:
            index = []
            indexes = [index for index, element in enumerate(nonzero_indices[1]) if element == item]
            index.append(indexes)
            index = index[0]
            if(len(index) > 1):
                #此index为y_pred中的索引
                index = nonzero_indices[0][index]
                x += torch.sum(y_pred[:, index[1]:index[-1]+1,int(item)] - y_pred[:, index[0]:index[-1], int(item)])        
        # x = y_pred[:, 1:, :] - y_pred[:, :-1, :]  # horizontal direction
        # y = y_pred[:, :, 1:] - y_pred[:, :, :-1]  # vertical direction

        # delta_x = x[:, :, 1:] ** 2
        # delta_y = y[:, 1:, :] ** 2
        delta_x = x ** 2
        delta_y = y ** 2
        delta_u = torch.abs(delta_x + delta_y)

        epsilon = 1e-8  # parameter to avoid square root of zero
        w = 0.1
        #length = w * torch.sum(torch.sqrt(delta_u + epsilon), dim=[1, 2])
        length = w * torch.sum(torch.sqrt(delta_u + epsilon))

        #area = torch.sum(y_pred, dim=[1, 2])
        area = torch.sum(y_pred[:,nonzero_indices[0],nonzero_indices[1]])
    
        compactness_loss = torch.sum(length ** 2 / (area * 4 * 3.1415926))
        
        return compactness_loss, torch.sum(length), torch.sum(area), delta_u
    def get_compactness_cost_v2(self,y_pred,para,nonzero_indices,target_mask,slice_pred):
        # Assuming y_pred is a PyTorch tensor of shape BxHxWxC
        a = para[2]
        b = para[3]
        L = 3.14 * 2 * b + 4*(a-b)
        S = 3.14 * a * b
        
        y_pred = y_pred[..., 0]
        
        # nonzero_indices = torch.tensor(nonzero_indices).to("cuda")
        x_indexes_dict = {}
        y_indexes_dict = {}
        for i, item in enumerate(nonzero_indices[0]):
            if item not in x_indexes_dict:
                x_indexes_dict[item] = []
                x_indexes_dict[item].append(nonzero_indices[1][i])
            else:
                x_indexes_dict[item].append(nonzero_indices[1][i])

        for i, item in enumerate(nonzero_indices[0]):
            if item not in y_indexes_dict:
                y_indexes_dict[item] = []
                y_indexes_dict[item].append(nonzero_indices[0][i])
            else:
                y_indexes_dict[item].append(nonzero_indices[0][i])

        y = 0
        x = 0
        for item, indexes in x_indexes_dict.items():
            if len(indexes) > 1:
                y_index_slice = slice(indexes[1], indexes[-1] + 1)
                y += torch.sum(y_pred[:, int(item), y_index_slice] - y_pred[:, int(item), indexes[0]:indexes[-1]])
        for item, indexes in y_indexes_dict.items():
            if len(indexes) > 1:
                x_index_slice = slice(indexes[1], indexes[-1] + 1)
                x += torch.sum(y_pred[:, x_index_slice, int(item)] - y_pred[:, indexes[0]:indexes[-1], int(item)]) 
        y = 0
        x = 0

        for item, indexes in x_indexes_dict.items():
            if len(indexes) > 1:
                y_index_slice = slice(indexes[1], indexes[-1] + 1)
                y += torch.sum(target_mask[:, int(item), y_index_slice] - target_mask[:, int(item), indexes[0]:indexes[-1]])
        for item, indexes in y_indexes_dict.items():
            if len(indexes) > 1:
                x_index_slice = slice(indexes[1], indexes[-1] + 1)
                x += torch.sum(target_mask[:, x_index_slice, int(item)] - target_mask[:, indexes[0]:indexes[-1], int(item)])    
        
        delta_x = x ** 2
        delta_y = y ** 2
        delta_u = delta_x + delta_y


        x = slice_pred[:, 1:, :] - slice_pred[:, :-1, :]  # horizontal direction
        y = slice_pred[:, :, 1:] - slice_pred[:, :, :-1]  # vertical direction

        delta_x = x[:, :, 1:] ** 2
        delta_y = y[:, 1:, :] ** 2
        delta_u = torch.abs(delta_x + delta_y)


        epsilon = 1e-8  # parameter to avoid square root of zero
        w = 1
        length = w * torch.sum(torch.sqrt(delta_u + epsilon), dim=[1, 2])
        #length = w * torch.sqrt(delta_u + epsilon)

        #area = torch.sum(y_pred, dim=[1, 2])
        area = torch.sum(y_pred[:,nonzero_indices[0],nonzero_indices[1]])
        ratio = L / (S + epsilon)
        #compactness_loss = torch.sum(length ** 2 / (area * 4 * 3.1415926))
        #compactness_loss = self.loss(length/(area+ epsilon),ratio) 
        #compactness_loss = torch.abs(length/(area+ epsilon) - ratio)
        compactness_loss = 1 - 1 / (1 + torch.log((length/(area+ epsilon)) / ratio))
        #compactness_loss = math.e**(-torch.log(length/(area+ epsilon)) / torch.log(ratio) + epsilon)
        #compactness_loss = self.sig(compactness_loss)
        return compactness_loss, torch.sum(length), torch.sum(area), delta_u

def wasserstein_distance_sigma(sigma1, sigma2):
    wasserstein_distance_item2 = torch.matmul(sigma1, sigma1) + torch.matmul(sigma2,
                                                                             sigma2) - 2 * sqrt_newton_schulz_autograd(
        torch.matmul(torch.matmul(sigma1, torch.matmul(sigma2, sigma2)), sigma1), 20, torch.FloatTensor)
    wasserstein_distance_item2 = trace(wasserstein_distance_item2)

    return wasserstein_distance_item2


# @weighted_loss
def gwds_loss(pred, target, weight, eps=1e-6):
    """IoU loss.
    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.
    Args:
        pred (Tensor): Predicted bboxes of format (xc, yc, w, h, a),
            shape (n, 5).
        target (Tensor): Corresponding gt bboxes, shape (n, 5).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    mask = (weight > 0).detach()
    pred = pred[mask]
    target = target[mask]

    x1 = pred[:, 0]
    y1 = pred[:, 1]
    w1 = pred[:, 2]
    h1 = pred[:, 3]
    theta1 = pred[:, 4]

    sigma1_1 = w1 / 2 * torch.cos(theta1) ** 2 + h1 / 2 * torch.sin(theta1) ** 2
    sigma1_2 = w1 / 2 * torch.sin(theta1) * torch.cos(theta1) - h1 / 2 * torch.sin(theta1) * torch.cos(theta1)
    sigma1_3 = w1 / 2 * torch.sin(theta1) * torch.cos(theta1) - h1 / 2 * torch.sin(theta1) * torch.cos(theta1)
    sigma1_4 = w1 / 2 * torch.sin(theta1) ** 2 + h1 / 2 * torch.cos(theta1) ** 2
    sigma1 = torch.reshape(
        torch.cat((sigma1_1.unsqueeze(1), sigma1_2.unsqueeze(1), sigma1_3.unsqueeze(1), sigma1_4.unsqueeze(1)), axis=1),
        (-1, 2, 2))

    x2 = target[:, 0]
    y2 = target[:, 1]
    w2 = target[:, 2]
    h2 = target[:, 3]
    theta2 = target[:, 4]
    sigma2_1 = w2 / 2 * torch.cos(theta2) ** 2 + h2 / 2 * torch.sin(theta2) ** 2
    sigma2_2 = w2 / 2 * torch.sin(theta2) * torch.cos(theta2) - h2 / 2 * torch.sin(theta2) * torch.cos(theta2)
    sigma2_3 = w2 / 2 * torch.sin(theta2) * torch.cos(theta2) - h2 / 2 * torch.sin(theta2) * torch.cos(theta2)
    sigma2_4 = w2 / 2 * torch.sin(theta2) ** 2 + h2 / 2 * torch.cos(theta2) ** 2
    sigma2 = torch.reshape(
        torch.cat((sigma2_1.unsqueeze(1), sigma2_2.unsqueeze(1), sigma2_3.unsqueeze(1), sigma2_4.unsqueeze(1)), axis=1),
        (-1, 2, 2))

    wasserstein_distance_item1 = (x1 - x2) ** 2 + (y1 - y2) ** 2
    wasserstein_distance_item2 = wasserstein_distance_sigma(sigma1, sigma2)
    wasserstein_distance = torch.max(wasserstein_distance_item1 + wasserstein_distance_item2,
                                     Variable(torch.zeros(wasserstein_distance_item1.shape[0]).type(torch.FloatTensor).cuda(),
                                              requires_grad=False))
    wasserstein_distance = torch.max(torch.sqrt(wasserstein_distance + eps),
                                     Variable(torch.zeros(wasserstein_distance_item1.shape[0]).type(torch.FloatTensor).cuda(),
                                              requires_grad=False))
    wasserstein_similarity = 1 / (wasserstein_distance + 2)
    wasserstein_loss = 1 - wasserstein_similarity

    return wasserstein_loss


def xywhr2xyrs(xywhr):
    xywhr = xywhr.reshape(-1, 5)
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7)
    r = torch.deg2rad(xywhr[..., 4])
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)
    return xy, R, S

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

def gwd_loss(pred, target, fun='log', tau=1.0, alpha=1.0, normalize=False):
    """
    given any positive-definite symmetrical 2*2 matrix Z:
    Tr(Z^(1/2)) = sqrt(λ_1) + sqrt(λ_2)
    where λ_1 and λ_2 are the eigen values of Z
    meanwhile we have:
    Tr(Z) = λ_1 + λ_2
    det(Z) = λ_1 * λ_2
    combination with following formula:
    (sqrt(λ_1) + sqrt(λ_2))^2 = λ_1 + λ_2 + 2 * sqrt(λ_1 * λ_2)
    yield:
    Tr(Z^(1/2)) = sqrt(Tr(Z) + 2 * sqrt(det(Z)))
    for gwd loss the frustrating coupling part is:
    Tr((Σp^(1/2) * Σt * Σp^(1/2))^(1/2))
    assuming Z = Σp^(1/2) * Σt * Σp^(1/2) then:
    Tr(Z) = Tr(Σp^(1/2) * Σt * Σp^(1/2))
    = Tr(Σp^(1/2) * Σp^(1/2) * Σt)
    = Tr(Σp * Σt)
    det(Z) = det(Σp^(1/2) * Σt * Σp^(1/2))
    = det(Σp^(1/2)) * det(Σt) * det(Σp^(1/2))
    = det(Σp * Σt)
    and thus we can rewrite the coupling part as:
    Tr((Σp^(1/2) * Σt * Σp^(1/2))^(1/2))
    = Tr{Z^(1/2)} = sqrt(Tr(Z) + 2 * sqrt(det(Z))
    = sqrt(Tr(Σp * Σt) + 2 * sqrt(det(Σp * Σt)))
    """
    xy_p, R_p, S_p = xywhr2xyrs(pred)
    xy_t, R_t, S_t = xywhr2xyrs(target)
    

    Sigma_p = R_p.matmul(S_p.square()).matmul(R_p.permute(0, 2, 1))
    Sigma_t = R_t.matmul(S_t.square()).matmul(R_t.permute(0, 2, 1))


    #kld xy
    Sigma_t_inv = torch.stack((Sigma_t[..., 1, 1], -Sigma_t[..., 0, 1],
                               -Sigma_t[..., 1, 0], Sigma_t[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    
    #求逆没问题
    Sigma_t_inv = Sigma_t_inv / (Sigma_t.det() ).unsqueeze(-1).unsqueeze(-1)

    Sigma_p_inv = torch.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                               -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    Sigma_p_inv = Sigma_p_inv / (Sigma_p.det()+1e-8).unsqueeze(-1).unsqueeze(-1)


    

    dxy = (xy_p - xy_t).unsqueeze(-1)
    #Dkl(p||t)
    #xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(Sigma_t_inv).bmm(dxy).view(-1)
    #Dkl(t||p)


    #gwd_xy
    #xy_distance = (xy_p - xy_t).square().sum(dim=-1)
    #kld xy
    xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(Sigma_t_inv).bmm(dxy).view(-1)
    #kld xy



    whr_distance = S_p.diagonal(dim1=-2, dim2=-1).square().sum(dim=-1)
    whr_distance = whr_distance + S_t.diagonal(dim1=-2, dim2=-1).square().sum(
        dim=-1)
    _t = Sigma_p.matmul(Sigma_t)

    _t_tr = _t.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = S_p.diagonal(dim1=-2, dim2=-1).prod(dim=-1)
    _t_det_sqrt = _t_det_sqrt * S_t.diagonal(dim1=-2, dim2=-1).prod(dim=-1)
    whr_distance = whr_distance + (-2) * ((_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt())

    distance = (xy_distance + alpha * alpha * whr_distance).clamp(0)
    # distance = (xy_distance + alpha * alpha * whr_distance).clamp(0).sqrt()

    if normalize:
        wh_p = pred[..., 2:4].clamp(min=1e-7, max=1e7)
        wh_t = target[..., 2:4].clamp(min=1e-7, max=1e7)
        scale = ((wh_p.log() + wh_t.log()).sum(dim=-1) / 4).exp()
        distance = distance / scale

    if fun == 'log':
        distance = torch.log1p(distance)
    elif fun == 'sqrt':
        distance = torch.sqrt(distance)
    else:
        raise ValueError('Invalid non-linear function {fun} for gwd loss')

    if tau >= 1.0:
        return 1 - 1 / (tau + distance)
    else:
        return distance
    




def kld_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    """Kullback-Leibler Divergence loss.
    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        sqrt (bool): Whether to sqrt the distance. Defaults to True.
    Returns:
        loss (torch.Tensor)
    """

    #R旋转矩阵 S比例矩阵
    xy_p, R_p, S_p = xywhr2xyrs(pred)
    xy_t, R_t, S_t = xywhr2xyrs(target)
    
    Sigma_p = R_p.matmul(S_p.square()).matmul(R_p.permute(0, 2, 1))
    Sigma_t = R_t.matmul(S_t.square()).matmul(R_t.permute(0, 2, 1))

    _shape = xy_p.shape

    # xy_p = xy_p.reshape(-1, 2)
    # xy_t = xy_t.reshape(-1, 2)
    # Sigma_p = Sigma_p.reshape(-1, 2, 2)
    # Sigma_t = Sigma_t.reshape(-1, 2, 2)

    Sigma_p_inv = torch.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                               -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    Sigma_p_inv = Sigma_p_inv / (Sigma_p.det() ).unsqueeze(-1).unsqueeze(-1)


    Sigma_t_inv = torch.stack((Sigma_t[..., 1, 1], -Sigma_t[..., 0, 1],
                               -Sigma_t[..., 1, 0], Sigma_t[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    Sigma_t_inv = Sigma_t_inv / (Sigma_t.det() ).unsqueeze(-1).unsqueeze(-1)
    

    dxy = (xy_p - xy_t).unsqueeze(-1)

    xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(Sigma_p_inv).bmm(dxy).view(-1)
    #NaN 因为他
    whr_distance = 0.5 * Sigma_p_inv.bmm(Sigma_t).diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    Sigma_p_det_log = torch.abs(Sigma_p.det()).log()
    Sigma_t_det_log = torch.abs(Sigma_t.det()).log()
    whr_distance = whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log)
    whr_distance = whr_distance - 1
    distance = (xy_distance / (alpha * alpha) + whr_distance ).clamp(0)
    #distance = xy_distance
    if sqrt:
        distance = distance.clamp(1e-7).sqrt()

    distance = distance.reshape(_shape[:-1])

    if fun == 'log':
        distance = torch.log1p(distance)
    elif fun == 'sqrt':
        distance = torch.sqrt(distance.clamp(1e-7))
    elif fun == 'none':
        pass
    else:
        raise ValueError(f'Invalid non-linear function {fun}')

    if tau >= 1.0:
        return 1 - 1 / (tau + distance)
    else:
        return distance



class GWDLoss(nn.Module):
    def __init__(self):
        super(GWDLoss, self).__init__()

    def forward(self, pred_tensor, target_tensor):

        ##处理参数，调成gwd需要的格式
        ind = target_tensor['ind']
        mask = target_tensor['reg_mask']
        pred_ab = _transpose_and_gather_feat(pred_tensor['ab'], ind)
        mask_ab = mask.unsqueeze(2).expand_as(pred_ab).float()
        pred_ang = _transpose_and_gather_feat(pred_tensor['ang'], ind)
        mask_ang = mask.unsqueeze(2).expand_as(pred_ang).float()

        from predict import _topk
        K = 100
        _, inds, _, x, y = _topk(pred_tensor['hm'])
        pred_xy = torch.cat([x.reshape((-1, K, 1)), y.reshape((-1, K, 1))], dim=2)
        pred = torch.cat([pred_xy * mask_ab,
                          pred_ab * 2 * mask_ab,
                          (pred_ang - 90) * mask_ang
                          ],
                         dim=2)

        _, _, _, x, y = _topk(target_tensor['hm'])
        target_xy = torch.cat([x.reshape((-1, K, 1)), y.reshape((-1, K, 1))], dim=2)
        target = torch.cat([target_xy * mask_ab,
                            target_tensor['ab'] * 2 * mask_ab,
                            (target_tensor['ang'] - 90) * mask_ang
                            ],
                           dim=2)
        #return torch.sum(kld_loss(pred, target, fun='log', tau=1.0, alpha=1.0, sqrt=False)) / (mask.sum() + 1e-8)
        return torch.sum(gwd_loss(pred, target, fun='log', tau=1.0, alpha=1.0, normalize=False)) / (mask.sum() + 1e-8)


class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()

    def forward(self, pred, target):
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
        return loss

class WeightLoss(nn.Module):
    def __init__(self):
        super(WeightLoss, self).__init__()

    def forward(self, pred):
        target = torch.tensor(np.array([0.4, 0.3, 0.2, 0.1]*2, dtype=np.float32)).cuda()
        target = target.reshape((-1,4,1,1))
        loss = F.smooth_l1_loss(pred, target, reduction='sum')
        return loss

class CtdetLoss(torch.nn.Module):
    def __init__(self, loss_weight):
        super(CtdetLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.crit_offset = RegL1Loscs_offset()
        self.crit_ab = RegL1Loss()
        # self.crit_ang = RegL1Loss()
        self.crit_ang = RegL1Loss_ang()
        self.crit_iou = GWDLoss()
        self.crit_mask = MaskLoss()
        # self.comp_loss = Compactness_loss()
        self.loss_weight = loss_weight

    def forward(self, pred_tensor, target_tensor):
        hm_weight = self.loss_weight['hm_weight']
        ab_weight = self.loss_weight['ab_weight']
        reg_weight = self.loss_weight['reg_weight']
        ang_weight = self.loss_weight['ang_weight']
        iou_weight = self.loss_weight['iou_weight']
        mask_weight = self.loss_weight['mask_weight']
        hmTop_weight = self.loss_weight['hmTop_weight']
        comp_weight = self.loss_weight['comp_weight']
        hm_loss, ab_loss, off_loss, ang_loss, iou_loss, mask_loss,hmtop_loss,topoff_loss,comp_loss = 0, 0, 0, 0, 0, 0, 0, 0, 0



        #中点 
        pred_tensor['hm'] = torch.sigmoid(pred_tensor['hm']) 
        hm_loss += self.crit(pred_tensor['hm'], target_tensor['hm']) 
        

        
        
        #顶点
        #顶点
        pred_tensor['hm_top'] = torch.sigmoid(pred_tensor['hm_top'])
        #hmtop_loss += self.crit(pred_tensor['hm_top'], target_tensor['hm_top'])
        hmtop_loss += self.crit(pred_tensor['hm_top'], target_tensor['hm_top'])
        hmtop_loss = self.crit(pred_tensor['hm_top'], target_tensor['hm_top'])

        
        


        #间接监督，loss回传后有可能导致 offset预测出nan 
        #通过顶点+偏移计算中点
        hm_top = pred_tensor['hm_top']
        #batch 1 128 128
        # hm_topCenter = torch.zeros(hm_top.shape[0],1,hm_top.shape[2],hm_top.shape[3],device="cuda")
        # for i in range(hm_top.shape[0]) :
        #     #每张图选100个顶点
        #     # 获得 tensor 中最大的 100 个元素及其位置
        #     k=100
        #     values, indices = torch.topk(hm_top[i,0,:,:].flatten(), k=100)
        #     rows = indices//hm_top.shape[2] 
        #     cols = indices % hm_top.shape[3]
        #     for j in range(k) :
        #         new_row = int(rows[j]+reg_top[i][0][rows[j]][cols[j]])
        #         new_col = int(cols[j]+reg_top[i][1][rows[j]][cols[j]])
        #         hm_topCenter[i][0][new_row][new_col] = hm_top[i][0][rows[j]][cols[j]]

        
            #遍历法太慢了 
            # for j in range(hm_top.shape[2]):
            #     for k in range(hm_top.shape[3]):
            #         #新坐标索引
            #         new_i = int(i+reg_top[i][0][j][k])
            #         new_j = int(j+reg_top[i][1][j][k])
            #         hm_topCenter[i][0][new_i][new_j] = hm_top[i][0][j][k]*0.1
        

        
        
    




        if ang_weight > 0:
            # ang_loss += self.crit_ang(pred_tensor['ang'], target_tensor['reg_mask'],
            #                           target_tensor['ind'], target_tensor['ang'])

            ang_loss += self.crit_ang(pred_tensor['ang'], target_tensor['reg_mask'],
                                      target_tensor['ind'], target_tensor['ang'],
                                      pred_tensor['ab'])

        if ab_weight > 0:
            ab_loss += self.crit_ab(pred_tensor['ab'], target_tensor['reg_mask'],
                                    target_tensor['ind'], target_tensor['ab'])
        #中点偏移量
        if reg_weight > 0:
            off_loss += self.crit_reg(pred_tensor['reg'], target_tensor['reg_mask'],
                                      target_tensor['ind'], target_tensor['reg'])
            
        if iou_weight > 0:
            iou_loss += self.crit_iou(pred_tensor, target_tensor)
        if mask_weight > 0:
            #mask_loss += self.crit_mask(pred_tensor['edge'], target_tensor['mask'])
            mask_loss += dice_loss(pred_tensor['edge'], target_tensor['mask'])  
        # if comp_weight > 0:
        #     comp_loss =  0 * self.comp_loss(pred_tensor['edge'], target_tensor['param'], target_tensor['mask'])
            #comp_loss +=  self.comp_loss(pred_tensor['edge'], target_tensor['param'], target_tensor['mask'])
            
        
            
    
#iou_weight * iou_loss + hmTop_weight * hmtop_loss,\
        return hm_weight * hm_loss + ab_weight * ab_loss + \
               ang_weight * ang_loss + reg_weight * off_loss + \
               iou_weight * iou_loss + hmTop_weight * hmtop_loss+\
               mask_weight * mask_loss,\
               [(hm_weight * hm_loss).item(), (ab_weight * ab_loss).item(),
                (ang_weight * ang_loss).item(), (reg_weight * off_loss).item(),
                (iou_weight * iou_loss).item(),(hmTop_weight * hmtop_loss).item(),
                (mask_weight * mask_loss).item()
            ]
        
        
