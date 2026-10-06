"""Selected canonical dependencies; no historical training entry point."""
import torch
from torch import nn

EDGES=[(0,7),(1,5),(2,6),(3,6),(4,7),(2,1),(5,6),(3,2)]

class Quantum(nn.Module):
    """Analytic 8-qubit circuit, exact same RY/RZ/reupload gate sequence as Exp7.
    CUDA complex64; parity with PennyLane checked before any training.
    """
    def __init__(self):
        super().__init__();self.weights=nn.Parameter(torch.empty(2,8,2).uniform_(-.1,.1))
        bits=((torch.arange(256)[:,None] >> torch.arange(7,-1,-1))&1)
        self.register_buffer('z_sign',1.-2.*bits)
        for i,(c,t) in enumerate(EDGES):
            perm=torch.arange(256)^(((torch.arange(256)>>(7-c))&1)<<(7-t))
            self.register_buffer(f'perm{i}',perm)
    def ry(self,s,a,q):
        x=s.reshape(-1,2**q,2,2**(7-q)); v0=x[:,:,0,:];v1=x[:,:,1,:]
        c=torch.cos(a/2).reshape(-1,1,1);d=torch.sin(a/2).reshape(-1,1,1)
        return torch.stack([c*v0-d*v1,d*v0+c*v1],2).reshape(-1,256)
    def rz(self,s,a,q):
        x=s.reshape(-1,2**q,2,2**(7-q));v0=x[:,:,0,:];v1=x[:,:,1,:]
        return torch.stack([v0*torch.exp(-.5j*a),v1*torch.exp(.5j*a)],2).reshape(-1,256)
    def forward(self,z):
        s=torch.zeros((len(z),256),dtype=torch.complex64,device=z.device);s[:,0]=1
        for b in range(2):
            for q in range(8):s=self.ry(s,torch.pi*torch.tanh(z[:,q]),q)
            for q in range(8):
                s=self.ry(s,self.weights[b,q,0],q);s=self.rz(s,self.weights[b,q,1],q)
            for i in range(len(EDGES)):s=s[:,getattr(self,f'perm{i}')]
        return (s.real.square()+s.imag.square())@self.z_sign

class Compressor(nn.Module):
    def __init__(self,columns,mapping):
        super().__init__();self.indices=[[columns.index(c) for c in cs if c in columns] for cs in mapping.values()]
        self.layers=nn.ModuleList([nn.Sequential(nn.Linear(len(i),max(2,min(8,len(i)*2))),nn.ReLU(),nn.Linear(max(2,min(8,len(i)*2)),1)) if i else nn.Identity() for i in self.indices])
    def forward(self,x):
        return torch.cat([m(x[:,i]) if i else x.new_zeros((len(x),1)) for i,m in zip(self.indices,self.layers)],1)

class Head(nn.Module):
    def __init__(self,kind):
        super().__init__();self.transform=Quantum() if kind=='QSN2' else nn.Sequential(nn.Linear(8,16),nn.Tanh(),nn.Linear(16,8))
        self.head=nn.Sequential(nn.Linear(16,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,z):return self.head(torch.cat([z,self.transform(z)],1)).squeeze(-1)
