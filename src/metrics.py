"""고정 conf/IoU의 객체 수준 진단. AP는 Ultralytics 공식 validator로 계산."""
import numpy as np


def iou_matrix(a,b):
    a = np.asarray(a).reshape(-1,4)
    b = np.asarray(b).reshape(-1,4)
    lo = np.maximum(a[:,None,:2],b[None,:,:2])
    hi = np.minimum(a[:,None,2:],b[None,:,2:])
    wh = np.maximum(hi-lo,0)
    inter = wh[:,:,0]*wh[:,:,1]
    area_a = np.prod(np.maximum(a[:,2:]-a[:,:2],0),axis=1)
    area_b = np.prod(np.maximum(b[:,2:]-b[:,:2],0),axis=1)
    return inter/np.maximum(area_a[:,None]+area_b[None,:]-inter,1e-12)


def match(gt_boxes,gt_classes,predictions,nc,threshold=0.5,class_aware=False):
    """클래스 무관 IoU 내림차순 1:1 연결로 오분류까지 confusion에 포함."""
    cm = np.zeros((nc+1,nc+1),dtype=int)  # true rows, predicted columns
    ious = iou_matrix(gt_boxes,predictions[:,:4])
    if class_aware:
        ious = ious*(np.asarray(gt_classes)[:,None] == predictions[None,:,5])
    candidates = np.argwhere(ious >= threshold)
    order = sorted(candidates.tolist(),key=lambda pair:(-ious[tuple(pair)],pair[0],pair[1]))
    used_gt,used_pred,events = set(),set(),[]
    for g,p in order:
        if g in used_gt or p in used_pred:
            continue
        used_gt.add(g)
        used_pred.add(p)
        actual,predicted = int(gt_classes[g]),int(predictions[p,5])
        cm[actual,predicted] += 1
        if actual != predicted:
            events.append({"type":"marine_as_trash" if actual==1 and predicted==0 else "trash_as_marine" if actual==0 and predicted==1 else "misclassified",
                           "true_class":actual,"predicted_class":predicted,"gt_index":g,"prediction_index":p,"iou":float(ious[g,p])})
    for g,c in enumerate(gt_classes):
        if g not in used_gt:
            cm[int(c),nc] += 1
            events.append({"type":"missed_trash" if int(c)==0 else "missed_object","true_class":int(c),"predicted_class":nc,"gt_index":g,"prediction_index":None,"iou":None})
    for p,row in enumerate(predictions):
        if p not in used_pred:
            c = int(row[5])
            cm[nc,c] += 1
            events.append({"type":"background_as_trash" if c==0 else "false_positive","true_class":nc,"predicted_class":c,"gt_index":None,"prediction_index":p,"iou":None})
    return cm,events


def scores(cm,names):
    rows = []
    for c,name in enumerate(names):
        tp = int(cm[c,c])
        fp = int(cm[:,c].sum()-tp)
        fn = int(cm[c,:].sum()-tp)
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn) if tp+fn else None
        f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
        rows.append({"class":name,"tp":tp,"fp":fp,"fn":fn,"support":tp+fn,"precision":precision,"recall":recall,"f1":f1})
    return rows
