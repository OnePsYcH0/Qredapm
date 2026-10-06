"""Canonical cached models; adapters remove project paths and audit-file writes.

Core model class bodies preserve the executed implementation. This module does
not load clinical data, checkpoints or results and does not start training.
"""
import copy
import random
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import model_redapm as recovered_faithful
import model_exp7 as recovered_exp7
from quantum_components import Quantum, Compressor, Head

def seed(s=42):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True

class EncoderStub(nn.Module):
    def __init__(self):super().__init__();self.config=SimpleNamespace(hidden_size=768);self.probe=None
    def forward(self,input_ids,attention_mask):
        assert self.probe is not None,'BERT outputs must be supplied from verified cache'
        return SimpleNamespace(last_hidden_state=self.probe)

class NoLoad:
    @staticmethod
    def from_pretrained(*args,**kwargs):return EncoderStub()

recovered_faithful.AutoModel=NoLoad
recovered_exp7.AutoModel=NoLoad

class SafeQuantum(Quantum):
    def forward(self,z):
        with torch.autocast(device_type=z.device.type,enabled=False):
            value=super().forward(z.float())
        if not torch.isfinite(value).all():raise ValueError('Nonfinite quantum output; never silently impute')
        return value

class CompactHead(Head):
    def __init__(self,kind):
        nn.Module.__init__(self)
        with torch.random.fork_rng():
            self.transform=SafeQuantum() if kind=='QSN2' else nn.Sequential(nn.Linear(8,16),nn.Tanh(),nn.Linear(16,8))
        self.head=nn.Sequential(nn.Linear(16,16),nn.ReLU(),nn.Linear(16,1))

class CanonicalSN3(nn.Module):
    def __init__(self,kind,columns,mapping,drugdim,compressor=None,no_text=False,no_drug=False):
        super().__init__();self.kind=kind;self.no_text=no_text;self.no_drug=no_drug
        if kind=='FaithfulREDAPM':
            self.net=recovered_faithful.REDAPM('cache',len(columns),drugdim,struct_hidden_dim=256,text_projection_dim=256,fusion_hidden_dim=256,struct_num_layers=3,max_visits=12,visit_transformer_layers=2,visit_transformer_heads=4,fusion_transformer_layers=2,fusion_transformer_heads=4,dropout=.3)
        else:
            present_mapping={k:[c for c in cs if c in columns] for k,cs in mapping.items()}
            self.net=recovered_exp7.Exp7Clinical8ClassicalReplacement('cache',columns,present_mapping,drugdim,dropout=.3)
            self.net.clinical_compressor=copy.deepcopy(compressor)
            for p in self.net.clinical_compressor.parameters():p.requires_grad=False
            # Identical downstream16->256 projection initialization for both branches.
            self.net.clinical_token_projection=nn.Sequential(nn.Linear(16,256),nn.LayerNorm(256),nn.ReLU(),nn.Dropout(.3))
            with torch.random.fork_rng():
                self.transform=SafeQuantum() if kind=='QuantumSN3' else nn.Sequential(nn.Linear(8,16),nn.Tanh(),nn.Linear(16,8))
        if no_text:
            for name in ['text_projection','visit_positional_embedding','visit_encoder','visit_pooling','visit_token_projection']:
                for p in getattr(self.net,name).parameters():p.requires_grad=False
        if no_drug and kind!='FaithfulREDAPM':
            for p in self.net.drug_token_projection.parameters():p.requires_grad=False

    def forward(self,visits,visit_mask,features,drug_code):
        m=self.net;batch=len(features)
        if self.no_text:
            visit_tokens=features.new_zeros((batch,0,256));v_mask=visit_mask[:,:0]
        else:
            vr=m.text_projection(visits)
            pos=torch.arange(12,device=features.device).unsqueeze(0).expand(batch,-1)
            vt=m.visit_encoder(vr+m.visit_positional_embedding(pos),src_key_padding_mask=~visit_mask.bool())
            visit_tokens=m.visit_token_projection(vt);v_mask=visit_mask
        cls=m.fusion_cls.expand(batch,-1,-1)
        if self.kind=='FaithfulREDAPM':
            st=m.structured_token_projection(m.structured_encoder(features,drug_code)).unsqueeze(1)
            tokens=torch.cat([cls,visit_tokens,st],dim=1)
            mods=torch.cat([torch.zeros((batch,1),device=features.device,dtype=torch.long),torch.ones((batch,visit_tokens.shape[1]),device=features.device,dtype=torch.long),torch.full((batch,1),2,device=features.device,dtype=torch.long)],1)
            tail=1
        else:
            z=m.clinical_compressor(features);ct=m.clinical_token_projection(torch.cat([z,self.transform(z)],1)).unsqueeze(1)
            ts=[cls,visit_tokens,ct];ms=[torch.zeros((batch,1),device=features.device,dtype=torch.long),torch.ones((batch,visit_tokens.shape[1]),device=features.device,dtype=torch.long),torch.full((batch,1),2,device=features.device,dtype=torch.long)]
            if not self.no_drug:
                ts.append(m.drug_token_projection(drug_code).unsqueeze(1));ms.append(torch.full((batch,1),3,device=features.device,dtype=torch.long))
            tokens=torch.cat(ts,1);mods=torch.cat(ms,1);tail=1 if self.no_drug else 2
        padding=torch.cat([torch.zeros((batch,1),device=features.device,dtype=torch.bool),~v_mask.bool(),torch.zeros((batch,tail),device=features.device,dtype=torch.bool)],1)
        out=m.fusion_encoder(tokens+m.modality_embedding(mods),src_key_padding_mask=padding)
        if self.no_text:pooled=features.new_zeros((batch,256))
        else:pooled=recovered_faithful.masked_mean_pool(out[:,1:13],v_mask)
        return m.classifier(torch.cat([out[:,0],pooled],-1)).squeeze(-1)

def verify_implementations(columns,mapping,drugdim):
    seed(42);comp=Compressor(columns,mapping)
    x=torch.randn(4,len(columns));drug=torch.randn(4,drugdim);v=torch.randn(4,12,768);mask=torch.ones(4,12,dtype=torch.long);mask[0,7:]=0
    faithful=CanonicalSN3('FaithfulREDAPM',columns,mapping,drugdim).eval()
    faithful.net.text_encoder.probe=v.reshape(-1,1,768)
    with torch.no_grad():
        p=faithful(v,mask,x,drug)
        q=faithful.net(torch.zeros(4,12,1,dtype=torch.long),torch.ones(4,12,1,dtype=torch.long),mask,x,drug)['logits']
    err=float((p-q).abs().max());assert err<1e-6
    # Original Exp7 forward with a compressor wrapper that emits matched[z,T(z)].
    classical=CanonicalSN3('ClassicalSN3',columns,mapping,drugdim,comp).eval()
    original=copy.deepcopy(classical.net);original.text_encoder.probe=v.reshape(-1,1,768)
    class Matched(nn.Module):
        def __init__(self,c,t):super().__init__();self.c=c;self.t=t
        def forward(self,x):
            z=self.c(x);return torch.cat([z,self.t(z)],1)
    original.clinical_compressor=Matched(original.clinical_compressor,copy.deepcopy(classical.transform))
    with torch.no_grad():
        a=classical(v,mask,x,drug);b=original(torch.zeros(4,12,1,dtype=torch.long),torch.ones(4,12,1,dtype=torch.long),mask,x,drug)['logits']
    exp7err=float((a-b).abs().max());assert exp7err<1e-6
    seed(42);c=CanonicalSN3('ClassicalSN3',columns,mapping,drugdim,comp)
    seed(42);q=CanonicalSN3('QuantumSN3',columns,mapping,drugdim,comp)
    for k,t in c.net.state_dict().items():assert torch.equal(t,q.net.state_dict()[k]),k
    import pennylane as qml
    seed(42);quant=SafeQuantum();z=torch.randn(4,8,requires_grad=True)
    @qml.qnode(qml.device('default.qubit',wires=8),interface='torch',diff_method='backprop')
    def ref(z,w):
        for layer in range(2):
            qml.AngleEmbedding(torch.pi*torch.tanh(z),wires=range(8),rotation='Y')
            for k in range(8):qml.RY(w[layer,k,0],wires=k);qml.RZ(w[layer,k,1],wires=k)
            for i,j in recovered_exp7.PAIRWISE_CLINICAL_EDGES:qml.CNOT(wires=[i,j])
        return [qml.expval(qml.PauliZ(k)) for k in range(8)]
    a=quant(z);b=torch.stack(ref(z,quant.weights),1)
    ga=torch.autograd.grad(a.sum(),(z,quant.weights),retain_graph=True);gb=torch.autograd.grad(b.sum(),(z,quant.weights))
    qe=float((a-b).abs().max());ge=max(float((u-w).abs().max()) for u,w in zip(ga,gb));assert qe<2e-5 and ge<2e-5
    return dict(faithful_forward_error=err,exp7_forward_error=exp7err,quantum_forward_error=qe,quantum_input_and_weight_gradient_error=ge,matched_shared_initialization_equal=True,reference='model_redapm/model_exp7 plus PennyLane',test_labels_used=False)
