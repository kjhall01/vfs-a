"""Read-only TRiSK curl/divergence and uniform-PV tendency audit.

Diagnostic candidates below are matrix ablations, never production replacements.
Synthetic prescribed f makes globally uniform positive PV possible on a closed
sphere; this is an algebraic test, not an Earth initial condition or time run.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
import torch
from vfs import VoronoiMesh
from vfs.ocean import OceanColumn, OceanState
from vfs.ocean.conservative import ConstantDensityDynamics


def array(x):
    return x.detach().cpu().numpy()


def matrix_from_stencil(indices, weights, size):
    row = np.broadcast_to(np.arange(size)[:, None], indices.shape)
    valid = indices >= 0
    return sparse.coo_matrix((weights[valid], (row[valid], indices[valid])), shape=(size,size)).tocsr()


def signed_area(a, b, c):
    """Oriented spherical triangle area, independent of production area helper."""
    return 2*np.arctan2(np.sum(a*np.cross(b,c),axis=-1),
                       1+np.sum(a*b,axis=-1)+np.sum(b*c,axis=-1)+np.sum(c*a,axis=-1))


def operators(mesh):
    n,e,v=mesh.nCells,mesh.nEdges,mesh.nVertices
    ac,av,dc,dv=map(array,(mesh.cell_area,mesh.dual_area,mesh.dc_edge,mesh.dv_edge))
    ec,ev=map(array,(mesh.edgesOnCell,mesh.edgesOnVertex))
    valid=ec>=0
    D=sparse.coo_matrix(((array(mesh.edgeSignOnCell)*dv[np.maximum(ec,0)]/ac[:,None])[valid],
                        (np.broadcast_to(np.arange(n)[:,None],ec.shape)[valid],ec[valid])),shape=(n,e)).tocsr()
    C=sparse.coo_matrix(((array(mesh.edgeSignOnVertex)*dc[ev]/av[:,None]).ravel(),
                        (np.repeat(np.arange(v),3),ev.ravel())),shape=(v,e)).tocsr()
    Jraw=matrix_from_stencil(array(mesh.tan_stencil),
                            -array(mesh.thuburn_edge_sign)[:,None]*array(mesh.tan_weights_masked),e)
    Jstored=matrix_from_stencil(array(mesh.rotation_stencil),array(mesh.rotation_weights_masked),e)
    # J is +k cross, hence the physical Coriolis acceleration is -f J u.
    return D,C,Jraw,Jstored


def transfers(mesh):
    """Independent primal/dual kite intersections versus stored corner areas."""
    n,v=mesh.nCells,mesh.nVertices
    area=np.zeros_like(array(mesh.thuburn_corner_area))
    c,vert=map(array,(mesh.cell_xyz,mesh.vertex_xyz))
    for i in range(n):
        ne=int(mesh.nEdgesOnCell[i]); vs=array(mesh.verticesOnCell[i,:ne]); es=mesh.edgesOnCell[i,:ne]
        # True primal/dual crossing: generator midpoint, not Voronoi-edge midpoint.
        midpoint=c[array(mesh.cellsOnEdge[es])].sum(1)
        midpoint/=np.linalg.norm(midpoint,axis=1,keepdims=True)
        a=np.broadcast_to(c[i],midpoint.shape); b=vert[vs]
        area[i,:ne]=(signed_area(a,midpoint,b)+signed_area(a,b,np.roll(midpoint,-1,axis=0)))*mesh.radius**2
    def transfer(areas,shift=0):
        rows=[];cols=[];values=[]
        for i in range(n):
            ne=int(mesh.nEdgesOnCell[i]); vs=array(mesh.verticesOnCell[i,:ne])
            rows.extend(vs);cols.extend([i]*ne)
            values.extend(np.roll(areas[i,:ne],shift)/array(mesh.dual_area)[vs])
        return sparse.coo_matrix((values,(rows,cols)),shape=(v,n)).tocsr()
    stored=array(mesh.thuburn_corner_area)*mesh.radius**2
    return area,transfer(area),transfer(stored),transfer(stored,-1)


def path_rotation(mesh, fraction, *, shift):
    """Independent path-sum assembly; shift=0 reproduces the production port.

    shift=-1 assigns the previous fraction to each traversed slot. The
    comparison diagnoses the exit-vertex indexing, not a proposed runtime API.
    """
    rows=[];cols=[];vals=[]
    for i in range(mesh.nCells):
        ne=int(mesh.nEdgesOnCell[i]); es=array(mesh.edgesOnCell[i,:ne]); vs=array(mesh.verticesOnCell[i,:ne])
        signs=array(mesh.edgeSignOnCell[i,:ne])
        for t,target in enumerate(es):
            endpoint=int(np.flatnonzero(array(mesh.verticesOnEdge[target])==vs[t])[0])
            circ=float(mesh.thuburn_endpoint_sign[target,endpoint])
            for s,source in enumerate(es):
                if s==t: continue
                path=0.;slot=(s+1)%ne
                while True:
                    path+=fraction[i,(slot+shift)%ne]
                    if slot==t: break
                    slot=(slot+1)%ne
                rows.append(target);cols.append(source)
                vals.append(-circ*signs[s]*(path-.5)*float(mesh.dv_edge[source]/mesh.dc_edge[target]))
    return sparse.coo_matrix((vals,(rows,cols)),shape=(mesh.nEdges,mesh.nEdges)).tocsr()


def norm(m):
    return float(np.linalg.norm(m.data))


def relative(a,b):
    return norm(a-b)/max(norm(b),1e-300)


def rms(x,weights):
    return float(np.sqrt(np.sum(weights*np.square(x))/np.sum(weights)))


def uniform_pv(mesh,P,Jstored,Jcompatible,amplitude):
    """Call production rotation with f prescribed so (curl(u)+f)/P(h)=q0.

    f is fixed during the instantaneous tendency. Restores the buffer afterward.
    The full Bernoulli contribution has zero curl and cannot repair a defect.
    """
    H=1000.;U=1.;f0=1e-4;q0=f0/H
    c=mesh.cell_xyz
    h=H*(1+amplitude*c[:,2:3])
    u=mesh.grad(U*mesh.radius*(c[:,0:1]+.3*c[:,1:2]*c[:,2:3]))
    hv=P@array(h[:,0]);zeta=array(mesh.curl(u)[:,0])
    f=q0*hv-zeta
    column=OceanColumn(1,total_depth=H)
    state=OceanState(u_e=u,h=h,temp_h=10*h,salt_h=35*h,mesh=mesh,column=column)
    dyn=ConstantDensityDynamics(omega=1.)
    original=mesh.coriolis_unit_vertex.clone()
    try:
        mesh.coriolis_unit_vertex.copy_(torch.from_numpy(f))
        production=array(dyn.rotation_acceleration(state)[:,0])
        terms=dyn.tendency_terms(state)
        full=array(mesh.curl(terms['u_e']['rotation']+terms['u_e']['bernoulli'])[:,0])
    finally:
        mesh.coriolis_unit_vertex.copy_(original)
    he=array(mesh.cell_to_edge(h)[:,0]);velocity=array(u[:,0]);flux=he*velocity
    D,C,_,_=operators(mesh)
    hdot=P@(-D@flux);expected=q0*hdot
    qedge=(q0*hv)[array(mesh.verticesOnEdge)].mean(1)
    A=-.5*(sparse.diags(qedge)@Jstored+Jstored@sparse.diags(qedge))
    metric=array(dyn.energy.metric(state)[:,0])
    paired=.5*(A@velocity-(A.T@(metric*velocity))/metric)
    B=-.5*(sparse.diags(qedge)@Jcompatible+Jcompatible@sparse.diags(qedge))
    paired_compatible=.5*(B@velocity-(B.T@(metric*velocity))/metric)
    # Constant-PV flux form with two choices of J: diagnostic only.
    choices={'production':production,'unpaired_absolute_vorticity':A@velocity,
             'paired_absolute_vorticity_compatible_geometry':paired_compatible,
             'pv_flux_stored_geometry':-q0*(Jstored@flux),
             'pv_flux_compatible_geometry':-q0*(Jcompatible@flux)}
    result={}
    av=array(mesh.dual_area)
    for name,acc in choices.items():
        defect=C@acc-expected
        result[name]={'relative_vorticity_tendency_defect':rms(defect,av)/max(rms(expected,av),1e-300),
                      'pv_relative_rate_per_second':rms(defect/hv,av)/q0,
                      'normalized_rotation_work':float(np.dot(metric*velocity,acc))/max(float(np.sum(np.abs(metric*velocity*acc))),1e-300)}
    return {'thickness_amplitude':amplitude,'initial_pv_relative_max':float(np.max(np.abs((zeta+f)/hv/q0-1))),
            'production_pairing_relative_error':float(np.linalg.norm(production-paired)/max(np.linalg.norm(production),1e-300)),
            'full_vs_rotation_curl_relative':float(np.linalg.norm(full-C@production)/max(np.linalg.norm(C@production),1e-300)),
            'expected_vorticity_tendency_rms_s2':rms(expected,av),'variants':result}


def measure(mesh,label):
    D,C,Jraw,Jstored=operators(mesh)
    area,P,Pstored,Pshift=transfers(mesh)
    fraction=area/array(mesh.cell_area)[:,None]
    independent=path_rotation(mesh,array(mesh.thuburn_corner_fraction),shift=0)
    index_only=path_rotation(mesh,array(mesh.thuburn_corner_fraction),shift=-1)
    compatible=path_rotation(mesh,fraction,shift=-1)
    target=P@D; metric=array(mesh.dc_edge*mesh.dv_edge)
    av,ac=map(array,(mesh.dual_area,mesh.cell_area))
    matrices={}
    for name,J in [('raw_port',Jraw),('stored_skew',Jstored),('index_only_ablation',index_only),('index_and_kite_ablation',compatible)]:
        weighted=sparse.diags(metric)@J
        matrices[name]={'curl_div_relative_frobenius':relative(C@J,target),
                        'metric_skew_relative_frobenius':norm(weighted+weighted.T)/max(norm(weighted),1e-300)}
    q=torch.randn(mesh.nCells,3,dtype=torch.float64,generator=torch.Generator().manual_seed(72))
    cg=mesh.curl(mesh.grad(q))*mesh.dual_area[:,None]
    return {'label':label,'cells':mesh.nCells,'edges':mesh.nEdges,'vertices':mesh.nVertices,
            'generator_sha256':hashlib.sha256(array(mesh.cell_xyz).tobytes()).hexdigest(),
            'geometry':{'true_kite_min_area':float(area[area!=0].min()),
                        'true_transfer_constant_max':float(np.max(np.abs(P@np.ones(mesh.nCells)-1))),
                        'true_transfer_mass_relative_max':float(np.max(np.abs(P.T@av/ac-1))),
                        'stored_transfer_constant_max':float(np.max(np.abs(Pstored@np.ones(mesh.nCells)-1))),
                        'stored_transfer_mass_relative_max':float(np.max(np.abs(Pstored.T@av/ac-1))),
                        'stored_fraction_partition_max':float(np.max(np.abs(array(mesh.thuburn_corner_fraction_sum)-1)))},
            'curl_grad_max_circulation':float(cg.abs().max()),
            'independent_raw_rebuild_relative':relative(independent,Jraw),
            'projection_change_relative':relative(Jstored,Jraw),
            'raw_vs_stored_corner_defect':relative(C@Jraw,Pstored@D),
            'raw_vs_shifted_corner_defect':relative(C@Jraw,Pshift@D),
            'index_only_vs_stored_corner_defect':relative(C@index_only,Pstored@D),
            'operators':matrices,
            'uniform_pv':[uniform_pv(mesh,P,Jstored,compatible,a) for a in (0.,.2)]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--generator-npy',type=Path,help='Audit only these frozen generator points')
    args=parser.parse_args()
    cases=[('fib64-0',lambda:VoronoiMesh(n_cells=64,lloyd_iters=0)),
           ('fib64-20',lambda:VoronoiMesh(n_cells=64,lloyd_iters=20))]
    if not args.quick:
        cases += [('fib256-20',lambda:VoronoiMesh(n_cells=256,lloyd_iters=20)),
                  ('fib1024-20',lambda:VoronoiMesh(n_cells=1024,lloyd_iters=20)),
                  ('fib4096-20',lambda:VoronoiMesh(n_cells=4096,lloyd_iters=20)),
                  ('icosa-f4',lambda:VoronoiMesh.refined_icosahedron(4)),
                  ('icosa-f8',lambda:VoronoiMesh.refined_icosahedron(8))]
    if args.generator_npy:
        xyz=np.load(args.generator_npy,allow_pickle=False)
        cases=[('frozen-'+str(len(xyz)),lambda:VoronoiMesh(n_cells=len(xyz),lloyd_iters=0,generator_xyz=xyz))]
    result={'torch':torch.__version__,'dtype':'float64','device':'cpu',
            'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'instantaneous operators; synthetic prescribed f; no modified model rollout', 'cases':[]}
    if args.generator_npy:
        result['input_npy_sha256']=hashlib.sha256(args.generator_npy.read_bytes()).hexdigest()
    for label,build in cases:
        result['cases'].append(measure(build().to(dtype=torch.float64),label))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(label,result['cases'][-1]['operators'],flush=True)

if __name__=='__main__':main()
