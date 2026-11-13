"""Ocean-owned horizontal volume/content ledger; no atmospheric imports.

This is a transport primitive, NOT an energy-closing active-ocean stepper.
It reuses the atmospheric design (carrier amounts, donor reference, all-face
correction budgets, authoritative Euler amounts) with ocean layer geometry.
T/S are independent concentrations, not species fractions summing to unity.
"""
from dataclasses import dataclass
import math

import torch


def divergence(i, j, transfer, nodes):
    """Net incoming amount on an oriented i->j graph; trailing field axes stay."""
    return transfer.new_zeros((nodes, *transfer.shape[1:])).index_add(0, i, -transfer).index_add(0, j, transfer)


@dataclass(frozen=True)
class TransportLedger:
    """Amounts are m3 times concentration; face transfers cover a FULL dt.

    First two fields are potential temperature and salinity; remaining fields
    are passive tracers. beta is per face AND field, unlike the atmospheric
    common species limiter. Bounds apply to this stage's incident neighbors.
    All tensors retain their graphs. No concentration is clipped after update.
    """
    volume: torch.Tensor
    amounts: torch.Tensor
    low_amounts: torch.Tensor
    integrated_flux: torch.Tensor
    high_integrated_flux: torch.Tensor
    donor_integrated_flux: torch.Tensor
    beta: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    i: torch.Tensor
    j: torch.Tensor
    carrier: torch.Tensor
    trace_change: torch.Tensor
    outflow_fraction: torch.Tensor

    def install(self, state, *, velocity=None):
        """Install authoritative amounts, not state + dt*diagnosed RHS."""
        area = state.mesh.cell_area[:, None]
        fields = self.amounts / area[..., None]
        return state.with_tensors(state.u_e if velocity is None else velocity,
            self.volume/area, fields[..., 0], fields[..., 1],
            None if state.tracer_h is None else fields[..., 2:])


class HorizontalTransport:
    """Conservative donor/FCT transport on independent horizontal layers.

    FCT = flux-corrected transport: limit a centered-flux correction to a donor
    reference. An actual-carrier outflow bound is enforced BEFORE any update.
    It is sufficient for positive remaining volume, not a gravity-wave CFL.
    No subcycling, retry, clipping, new heat reservoir or force correction.
    """
    def __init__(self, mesh, *, domain=None, scheme='fct', cfl_safety=.9, min_thickness=1e-3):
        if scheme not in ('donor', 'fct'):
            raise ValueError("scheme must be 'donor' or 'fct'")
        if not 0 < cfl_safety < 1 or not math.isfinite(min_thickness) or min_thickness < 0:
            raise ValueError('invalid carrier safety/minimum thickness')
        if domain is not None and domain.mesh is not mesh:
            raise ValueError('domain must borrow this mesh')
        self.mesh, self.domain, self.scheme = mesh, domain, scheme
        self.cfl_safety, self.min_thickness = cfl_safety, min_thickness

    def prepare(self, layers):
        """Precompute immutable layer connectivity; no state-dependent quantities.

        Rebuild the model after changing connectivity or device. Tensor versions
        reject in-place edits rather than silently using a stale graph.
        """
        cells = self.mesh.cellsOnEdge
        active = None if self.domain is None else self.domain.active_edge
        if cells.is_inference() or (active is not None and active.is_inference()):
            # Inference tensors have no mutation counter: preserve the ordinary
            # indexing path rather than cache geometry we cannot invalidate.
            self.__dict__.pop('_prepared', None)
            return
        edges = cells if active is None else cells[active]
        k = torch.arange(layers, device=cells.device)
        i = (edges[:, 0, None]*layers+k).flatten()
        j = (edges[:, 1, None]*layers+k).flatten()
        degree = self.mesh.cell_area.new_zeros(self.mesh.nCells*layers)
        ones = degree.new_ones(i.shape)
        degree = degree.index_add(0, i, ones).index_add(0, j, ones)
        self._prepared = dict(layers=layers, cells=cells, version=cells._version,
            active=active, active_version=None if active is None else active._version,
            i=i, j=j, neighbors=torch.cat((j,i)), degree=degree)

    def _indices(self, layers, device):
        p = getattr(self, '_prepared', None)
        if p is None:
            cells = self.mesh.cellsOnEdge
            if self.domain is not None: cells = cells[self.domain.active_edge]
            k = torch.arange(layers, device=device)
            return (cells[:,0,None]*layers+k).flatten(), (cells[:,1,None]*layers+k).flatten()
        active = None if self.domain is None else self.domain.active_edge
        if (layers != p['layers'] or self.mesh.cellsOnEdge is not p['cells']
                or self.mesh.cellsOnEdge._version != p['version'] or p['i'].device != device
                or p['degree'].dtype != self.mesh.cell_area.dtype
                or active is not p['active'] or (active is not None and active._version != p['active_version'])):
            raise ValueError('prepared transport geometry changed; rebuild the model')
        return p['i'], p['j']

    def graph(self, state):
        state.validate()
        if state.mesh is not self.mesh or state.h.ndim != 2:
            raise ValueError('transport requires an unbatched state on its mesh')
        if not state.h.is_floating_point():
            raise ValueError('state must contain floating-point tensors')
        for value in (self.mesh.cell_area, self.mesh.dv_edge,
                      state.column.rest_thickness, state.column.total_depth):
            if value.dtype != state.h.dtype or value.device != state.h.device:
                raise ValueError('geometry must match state dtype/device; convert explicitly')
        for name in ('u_e', 'h', 'temp_h', 'salt_h', 'tracer_h'):
            value = getattr(state, name)
            if value is not None and (value.ndim != (3 if name == 'tracer_h' else 2)
                                      or value.dtype != state.h.dtype or value.device != state.h.device
                                      or not torch.isfinite(value).all().item()):
                raise ValueError('state tensors must be unbatched, finite and share dtype/device')
        state.column.validate_thickness(state.h,min_thickness=self.min_thickness)
        if (state.salt_h < 0).any().item():
            raise ValueError('initial salinity must be nonnegative')
        mesh, layers = self.mesh, state.h.shape[-1]
        flux = mesh.dv_edge[:,None]*mesh.cell_to_edge(state.h)*state.u_e
        if self.domain is not None:
            flux = flux[self.domain.active_edge]
        i, j = self._indices(layers, state.h.device)
        volume = (mesh.cell_area[:,None]*state.h).flatten()
        return volume,i,j,flux.flatten()

    def timestep_limit(self, state):
        """Carrier-only maximum dt, seconds; infinity for a stationary graph."""
        volume,i,j,flux = self.graph(state)
        out = torch.zeros_like(volume).index_add(0,i,flux.clamp_min(0)).index_add(0,j,(-flux).clamp_min(0))
        floor = (self.mesh.cell_area[:,None]*torch.full_like(state.h,self.min_thickness)).flatten()
        safe = torch.where(out > 0,out,torch.ones_like(out))
        return torch.where(out > 0,self.cfl_safety*(volume-floor)/safe,torch.full_like(out,float('inf'))).min()

    def advance(self, state, dt):
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        volume,i,j,flux = self.graph(state)
        G = float(dt)*flux
        outgoing = torch.zeros_like(volume).index_add(0,i,G.clamp_min(0)).index_add(0,j,(-G).clamp_min(0))
        floor = (self.mesh.cell_area[:,None]*torch.full_like(state.h,self.min_thickness)).flatten()
        if (outgoing > self.cfl_safety*(volume-floor)).any().item():
            raise FloatingPointError('actual carrier outflow exceeds Euler budget; reduce dt')
        remaining = volume-outgoing
        incoming = torch.zeros_like(volume).index_add(0,j,G.clamp_min(0)).index_add(0,i,(-G).clamp_min(0))
        new_volume = remaining+incoming
        fields = [state.temp_h[...,None],state.salt_h[...,None]]
        if state.tracer_h is not None:
            fields.append(state.tracer_h)
        q = (torch.cat(fields,-1)/state.h[...,None]).reshape(volume.numel(),-1)
        qi, qj = q[i], q[j]
        positive, negative = G.clamp_min(0), (-G).clamp_min(0)
        donor = torch.where((G >= 0)[:,None],qi,qj)
        centered = .5*(qi+qj)
        low_flux, high_flux = G[:,None]*donor, G[:,None]*centered
        # Positive remaining-carrier representation avoids subtracting nearly
        # equal scalar amounts. Incoming volume always carries the donor value.
        low = (remaining[:,None]*q).index_add(0,j,positive[:,None]*qi)
        low = low.index_add(0,i,negative[:,None]*qj)
        prepared = getattr(self, '_prepared', None)
        neighbors = (torch.cat((j,i)) if prepared is None else prepared['neighbors'])[:,None].expand(-1,q.shape[-1])
        values = torch.cat((qi,qj))
        qmin = q.scatter_reduce(0,neighbors,values,reduce='amin',include_self=True)
        qmax = q.scatter_reduce(0,neighbors,values,reduce='amax',include_self=True)
        lower, upper = new_volume[:,None]*qmin,new_volume[:,None]*qmax
        anti = high_flux-low_flux
        plus = torch.zeros_like(low).index_add(0,j,anti.clamp_min(0)).index_add(0,i,(-anti).clamp_min(0))
        minus = torch.zeros_like(low).index_add(0,i,anti.clamp_min(0)).index_add(0,j,(-anti).clamp_min(0))
        def ratio(budget,demand):
            denominator = torch.where(demand > 0,demand,torch.ones_like(demand))
            return torch.where(demand > 0,budget.clamp_min(0)/denominator,torch.ones_like(demand)).clamp(max=1)
        rp, rm = ratio(upper-low,plus),ratio(low-lower,minus)
        beta = torch.where(anti >= 0,torch.minimum(rm[i],rp[j]),torch.minimum(rp[i],rm[j]))
        if self.scheme == 'donor':
            beta = torch.zeros_like(beta)
        else:
            beta = torch.where(beta < 1,beta*(1-64*torch.finfo(q.dtype).eps),beta)
        amounts = low+divergence(i,j,beta*anti,volume.numel())
        tolerance = 256*torch.finfo(q.dtype).eps*new_volume[:,None]*q.abs().amax(0)
        if not torch.isfinite(amounts).all().item() or ((amounts < lower-tolerance)|(amounts > upper+tolerance)).any().item():
            raise FloatingPointError('bounded ledger failed its stage bounds')
        # Reject rather than repair negativity, including roundoff. Signed T
        # and signed passive fields are allowed; nonnegative inputs stay so.
        if ((amounts < 0) & (qmin >= 0)).any().item():
            raise FloatingPointError('negative concentration in bounded ledger')
        return TransportLedger(new_volume.reshape_as(state.h),
            amounts.reshape(*state.h.shape,-1),low.reshape(*state.h.shape,-1),
            low_flux+beta*anti,high_flux,low_flux,beta,lower,upper,i,j,flux,
            (1-beta)*(donor-centered),(outgoing/volume).reshape_as(state.h))
