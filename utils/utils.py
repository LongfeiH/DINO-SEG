def compute_miou(pred, label, num_classes=152):
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        label_inds = (label == cls)
        intersection = (pred_inds & label_inds).sum().item()
        union = (pred_inds | label_inds).sum().item()
        if union == 0:
            continue  # 忽略未出现的类
        ious.append(intersection / union)
    return sum(ious) / len(ious)
