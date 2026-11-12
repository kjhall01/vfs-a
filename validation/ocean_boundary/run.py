"""Optional CPU float64 f-plane audit of shared column kernels, not a dycore."""
import json
import numpy as np
import torch
from scipy.linalg import expm
from vfs.ocean.stratified.boundary import wind_velocity, drag_velocity
from vfs.ocean.stratified.tke import diffuse


def run():
    rho,nu,f,depth,tau=1025.,.01,1e-4,200.,.1
    def system(n):
        h=depth/n
        d=np.diff(np.eye(n),axis=0)
        lap=-nu/h**2*d.T@d
        a=np.block([[lap,f*np.eye(n)],[-f*np.eye(n),lap]])
        b=np.zeros(2*n);b[0]=tau/(rho*h)
        return h,a,b
    spatial=[]
    for n in (16,32,64,128):
        h,a,b=system(n)
        steady=np.linalg.solve(a,-b)
        lam=(1+1j)*np.sqrt(f/(2*nu))
        edges=np.linspace(0,-depth,n+1)
        reference=tau/(rho*nu*lam**2*np.sinh(lam*depth))*np.diff(-np.sinh(lam*(edges+depth)))/h
        actual=steady[:n]+1j*steady[n:]
        error=np.linalg.norm(actual-reference)/np.linalg.norm(reference)
        transport=h*actual.sum()
        assert abs(transport+1j*tau/(rho*f))<1e-12
        spatial.append(dict(layers=n,relative_profile_error=float(error),transport_real=float(transport.real),transport_imag=float(transport.imag)))
    assert all(b['relative_profile_error']<.3*a['relative_profile_error'] for a,b in zip(spatial,spatial[1:]))
    n=32;h,a,b=system(n);duration=21600.
    aug=np.zeros((2*n+1,2*n+1));aug[:-1,:-1]=a;aug[:-1,-1]=b
    reference=expm(aug*duration)[:-1,-1].reshape(2,n)
    reference_transport=h*(reference[0]+1j*reference[1]).sum()
    analytic_transport=tau/rho*(1-np.exp(-1j*f*duration))/(1j*f)
    assert abs(reference_transport-analytic_transport)<1e-12
    transient=[]
    for dt in (600.,300.,150.,75.):
        u=torch.zeros(2,n,dtype=torch.float64)
        mass=torch.full_like(u,h);g=torch.full((2,n-1),nu/h,dtype=u.dtype)
        work=0.;dissipation=0.
        for _ in range(round(duration/dt)):
            before=u.clone()
            top=wind_velocity(u[:,0],u.new_tensor([tau,0.]),h,rho,dt)
            u=torch.cat((top[:,None],u[:,1:]),1)
            work+=float(.5*rho*h*(u.square()-before.square()).sum())
            before=u;u=diffuse(u,mass,g,dt)
            dissipation+=float(.5*rho*h*(before.square()-u.square()).sum())
            c,s=np.cos(f*dt),np.sin(f*dt)
            u=torch.stack((c*u[0]+s*u[1],-s*u[0]+c*u[1]))
        error=float(np.linalg.norm(u.numpy()-reference)/np.linalg.norm(reference))
        residual=float(.5*rho*h*u.square().sum())+dissipation-work
        transient.append(dict(dt=dt,relative_profile_error=error,work_J_m2=work,diffusive_loss_J_m2=dissipation,energy_residual_J_m2=residual))
    assert all(b['relative_profile_error']<.55*a['relative_profile_error'] for a,b in zip(transient,transient[1:]))
    assert max(abs(x['energy_residual_J_m2']) for x in transient)<1e-10
    u=torch.tensor([.3,.4],dtype=torch.float64);initial=u.clone()
    for _ in range(100):u=drag_velocity(u,u.norm(),20.,.0025,10.)
    exact=initial/(1+1000*.0025*initial.norm()/20.)
    drag_error=float((u-exact).norm()/exact.norm())
    assert drag_error<1e-14
    return dict(scope='Shared wind/drag/diffusion kernels; independent constant-viscosity f-plane references, not evolving-TKE spherical Ekman qualification',rho0=rho,viscosity=nu,coriolis=f,depth=depth,stress=tau,steady=spatial,transient=transient,drag_relative_error=drag_error)

if __name__=='__main__':
    print(json.dumps(run(),indent=2))
