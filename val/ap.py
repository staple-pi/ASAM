import numpy as np

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    else:
        return intersection / union

def match_predictions(pred_masks, pred_scores, true_masks, iou_threshold=0.5):
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_masks = [pred_masks[i] for i in sorted_indices]
    pred_scores = [pred_scores[i] for i in sorted_indices]
    
    matched_true = set()
    TPs = []
    FPs = []
    
    for pred_mask in pred_masks:
        best_iou = 0
        best_true = -1
        for idx, true_mask in enumerate(true_masks):
            if idx in matched_true:
                continue
            iou = calculate_iou(pred_mask, true_mask)
            if iou > best_iou:
                best_iou = iou
                best_true = idx
        if best_iou >= iou_threshold:
            TPs.append(1)
            FPs.append(0)
            matched_true.add(best_true)
        else:
            TPs.append(0)
            FPs.append(1)
    
    FNs = len(true_masks) - len(matched_true)
    
    return TPs, FPs, FNs

def compute_precision_recall(TPs, FPs, FNs):
    precisions = []
    recalls = []
    cumulative_TP = 0
    cumulative_FP = 0
    for tp, fp in zip(TPs, FPs):
        cumulative_TP += tp
        cumulative_FP += fp
        precision = cumulative_TP / (cumulative_TP + cumulative_FP + 1e-6)
        recall = cumulative_TP / (cumulative_TP + FNs + 1e-6)
        precisions.append(precision)
        recalls.append(recall)
    return precisions, recalls

def calculate_ap(precisions, recalls):
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_indices = np.argsort(recalls)
    precisions = precisions[sorted_indices]
    recalls = recalls[sorted_indices]
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i-1]) * precisions[i]
    return ap

def evaluate_ap(pred_masks, pred_scores, true_masks, iou_threshold):
    TPs, FPs, FNs = match_predictions(pred_masks, pred_scores, true_masks, iou_threshold)
    precisions, recalls = compute_precision_recall(TPs, FPs, FNs)
    ap = calculate_ap(precisions, recalls)
    return ap

def evaluate_all_aps(pred_masks, pred_scores, true_masks):
    ap = evaluate_ap(pred_masks, pred_scores, true_masks, iou_threshold=0.5)
    ap50 = evaluate_ap(pred_masks, pred_scores, true_masks, iou_threshold=0.5)
    ap75 = evaluate_ap(pred_masks, pred_scores, true_masks, iou_threshold=0.75)
    return {'AP': ap, 'AP50': ap50, 'AP75': ap75}

# 示例数据
# 假设有3个预测掩码和2个真实掩码
pred_masks = [
    np.array([[1, 1], [0, 0]]),  # 预测1
    np.array([[1, 0], [1, 0]]),  # 预测2
    np.array([[0, 1], [0, 1]])   # 预测3
]
pred_scores = [0.9, 0.75, 0.6]  # 预测得分

true_masks = [
    np.array([[1, 1], [0, 0]]),  # 真实1
    np.array([[0, 1], [0, 1]])   # 真实2
]

# 计算 AP, AP50, AP75
aps = evaluate_all_aps(pred_masks, pred_scores, true_masks)
print("AP:", aps['AP'])
print("AP50:", aps['AP50'])
print("AP75:", aps['AP75'])
