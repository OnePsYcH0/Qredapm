"""Input construction functions; no patient records or model weights."""

def text_document(row):
    def flat(x):return ' '.join(str(i).strip() for i in (x or []) if str(i).strip())
    return f"疾病信息：{flat(row.get('disease_names'))}。就诊记录：{flat(row.get('visit_sn'))}"

def visit_texts(row):
    a=row.get('visit_sn') or [];b=row.get('disease_names') or []
    values=[f"诊断信息：{b[i] if i<len(b) else ''}。就诊记录：{a[i] if i<len(a) else ''}" for i in range(min(12,max(len(a),len(b))))]
    if not values: values=['诊断信息：。就诊记录：'] # Explicit empty-history token, avoid all-masked attention NaNs.
    mask=[1]*len(values)+[0]*(12-len(values))
    return values+['诊断信息：。就诊记录：']*(12-len(values)),mask
