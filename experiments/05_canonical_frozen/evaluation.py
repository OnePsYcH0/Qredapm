"""Metric and threshold functions; callers supply their own authorized inputs.

No clinical data, predictions or precomputed results are embedded in this module.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.linear_model import LogisticRegression

def choose_threshold(y,p,rule='f1',target=.70):
    y=np.asarray(y);p=np.asarray(p)
    if rule=='f1':
        prec,rec,ts=precision_recall_curve(y,p)
        f=2*prec[:-1]*rec[:-1]/np.maximum(prec[:-1]+rec[:-1],1e-15)
        # Deterministic highest threshold among exact best-F1 ties.
        return float(ts[np.flatnonzero(f==f.max())[-1]])
    fpr,tpr,ts=roc_curve(y,p,drop_intermediate=False)
    eligible=np.where(tpr>=target)[0]
    return float(ts[eligible[0]])

def metrics(y,p,t):
    y=np.asarray(y,dtype=int);p=np.asarray(p,float); pred=p>=t
    tp=np.sum((y==1)&pred); tn=np.sum((y==0)&~pred)
    fp=np.sum((y==0)&pred); fn=np.sum((y==1)&~pred)
    div=lambda a,b:float(a/b) if b else np.nan
    se=div(tp,tp+fn);sp=div(tn,tn+fp)
    return dict(roc_auc=float(roc_auc_score(y,p)),pr_auc_ap=float(average_precision_score(y,p)),
        f1=div(2*tp,2*tp+fp+fn),accuracy=div(tp+tn,len(y)),balanced_accuracy=(se+sp)/2,
        sensitivity=se,specificity=sp,ppv=div(tp,tp+fp),npv=div(tn,tn+fn),
        lr_positive=div(se,1-sp),lr_negative=div(1-se,sp),brier=float(np.mean((p-y)**2)),
        threshold=float(t),n=len(y),positive_rate=float(y.mean()),tp=int(tp),tn=int(tn),fp=int(fp),fn=int(fn))

B=2000;BOOT_SEED=20261006

M=['roc_auc','pr_auc_ap','f1','accuracy','balanced_accuracy','sensitivity','specificity','ppv','npv','lr_positive','lr_negative','brier']

def boot_metrics(y,p,t,weights):
    """Weighted empirical metrics; repeated scores are grouped (including AP ties).
    Each weight row is exactly an ordinary cluster bootstrap resample.
    """
    order=np.argsort(-p,kind='stable');ys=y[order];ps=p[order]
    ends=np.r_[np.flatnonzero(np.diff(ps)!=0),len(p)-1];pred=p>=t;out=[]
    with np.errstate(divide='ignore',invalid='ignore'):
        for w in np.array_split(weights,20):
            ws=w[:,order].astype(float);tp=np.cumsum(ws*ys,axis=1)[:,ends];fp=np.cumsum(ws*(1-ys),axis=1)[:,ends]
            totalp=tp[:,-1];totaln=fp[:,-1];rec=tp/totalp[:,None];fpr=fp/totaln[:,None]
            delta_rec=np.diff(np.column_stack([np.zeros(len(w)),rec]),axis=1)
            precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)
            ap=(delta_rec*precision).sum(1)
            rprev=np.column_stack([np.zeros(len(w)),rec[:,:-1]])
            a=((rec+rprev)*.5*np.diff(np.column_stack([np.zeros(len(w)),fpr]),axis=1)).sum(1)
            TP=w@((y==1)&pred);FP=w@((y==0)&pred);TN=w@((y==0)&~pred);FN=w@((y==1)&~pred)
            se=TP/(TP+FN);sp=TN/(TN+FP)
            out.append(np.column_stack([a,ap,2*TP/(2*TP+FP+FN),(TP+TN)/w.sum(1),(se+sp)/2,se,sp,TP/(TP+FP),TN/(TN+FN),se/(1-sp),(1-se)/sp,(w@((p-y)**2))/w.sum(1)]))
    return np.concatenate(out)

def cluster_weights(ids):
    unique,inv=np.unique(ids,return_inverse=True);rng=np.random.default_rng(BOOT_SEED)
    counts=rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=B).astype(np.float32)
    return counts[:,inv]

def calibration(y,p):
    pclip=np.clip(p,1e-6,1-1e-6);z=np.log(pclip/(1-pclip)).reshape(-1,1)
    model=LogisticRegression(penalty=None,solver='lbfgs',max_iter=2000,tol=1e-10);model.fit(z,y)
    bins=np.minimum((p*10).astype(int),9);rows=[];ece=0.
    for b in range(10):
        mask=bins==b
        if mask.any():
            rows.append(dict(bin=b,n=int(mask.sum()),mean_probability=float(p[mask].mean()),observed_rate=float(y[mask].mean())))
            ece+=mask.mean()*abs(p[mask].mean()-y[mask].mean())
    return dict(brier=float(np.mean((p-y)**2)),ece_10_equal_width=float(ece),calibration_intercept=float(model.intercept_[0]),calibration_slope=float(model.coef_[0,0]),calibration_fit='unpenalized diagnostic only; predictions unchanged'),rows
