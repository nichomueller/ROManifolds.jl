using Gridap
using GridapEmbedded
using ROManifolds
using ROManifolds.DofMaps
using ROManifolds.TProduct
using Test

import ROManifolds.DofMaps: get_dof_to_bg_dof, get_bg_dof_to_dof, get_mdof_to_dof

R = 0.3
pmin = Point(0,0)
pmax = Point(1,1)
n = 10
partition = (n,n)

dp = pmax - pmin

geo1 = disk(R,x0=Point(0.5,0.5))
geo2 = ! geo1

model = CartesianDiscreteModel(pmin,pmax,partition)
cutgeo = cut(model,geo2)

strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

Ωbg = Triangulation(model)
Ωact = Triangulation(cutgeo,ACTIVE)
Ωactout = Triangulation(cutgeo,ACTIVE_OUT)
Ω = Triangulation(cutgeo,PHYSICAL_IN)
Ωout = Triangulation(cutgeo,PHYSICAL_OUT)

order = 1
degree = 2*order

# Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
dΩout = Measure(Ωactout,degree)
Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)
Γ = EmbeddedBoundary(cutgeo)
n_Γ = get_normal_vector(Γ)
dΓ = Measure(Γ,degree)

f(x) = x[1]
h(x) = x[2]
g(x) = x[2]-x[1]

const γd = 10.0
const hd = dp[1]/n

a(u,v) = ∫(∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
l(v) =  ∫(f⋅v)dΩ + ∫(h⋅v)dΓn + ∫( (γd/hd)*v*g - (n_Γ⋅∇(v))*g )dΓ

reffe = ReferenceFE(lagrangian,Float64,2)
Vact = FESpace(Ωact,reffe,conformity=:H1)
Vagg = AgFEMSpace(Vact,aggregates)
Uagg = TrialFESpace(Vagg,g)

aout(u,v) = ∫(∇(v)⋅∇(u))dΩout
lout(v) =  ∫(∇(v)⋅∇(g))dΩout

Vout = FESpace(Ωactout,reffe,conformity=:H1)
Uout = TrialFESpace(Vout,g)

V = FESpace(model,reffe,conformity=:H1)

# act_dof_to_bgdof = get_dof_to_bg_dof(V,Vact)
# agg_dof_to_adof = get_mdof_to_dof(Vagg)
# agg_dof_to_bgdof = compose_index(agg_dof_to_adof,act_dof_to_bgdof)
agg_dof_to_bgdof =  get_dof_to_bg_dof(V,Vagg)
act_out_dof_to_bgdof = get_dof_to_bg_dof(V,Vout)
agg_out_dof_to_bgdof = setdiff(act_out_dof_to_bgdof,agg_dof_to_bgdof)
agg_out_dof_to_actdof = get_bg_dof_to_dof(V,Vout,agg_out_dof_to_bgdof)

op = AffineFEOperator(a,l,Uagg,Vagg)

gV = interpolate_everywhere(g,V)
gV_out = view(gV.free_values,agg_out_dof_to_bgdof)
ext = FunctionalExtension(gV_out,agg_out_dof_to_bgdof)
solver = ExtensionSolver(LUSolver(),ext)
u = solve(solver,op.op)
uh = FEFunction(V,u)

ext = ZeroExtension(agg_out_dof_to_bgdof)
solver = ExtensionSolver(LUSolver(),ext)
u = solve(solver,op.op)
uh = FEFunction(V,u)

A = assemble_matrix(aout,Uout,Vout)
b = assemble_vector(lout,Vout)
ext = HarmonicExtension(A,b,agg_out_dof_to_bgdof,agg_out_dof_to_actdof)
solver = ExtensionSolver(LUSolver(),ext)
u = solve(solver,op.op)
uh = FEFunction(V,u)

using DrWatson
writevtk(Ωbg,datadir("plts/sol0.vtu"),cellfields=["uh"=>uh])

CIAO
# function get_mdof_to_dof(f)
#   T = eltype(f.mDOF_to_DOF)
#   mdof_to_dof = zeros(T,f.n_fmdofs)
#   for mDOF in eachindex(mdof_to_dof)
#     DOF = f.mDOF_to_DOF[mDOF]
#     mdof = FESpaces._DOF_to_dof(mDOF,f.n_fmdofs)
#     dof = FESpaces._DOF_to_dof(DOF,f.n_fdofs)
#     if mdof > 0
#       mdof_to_dof[mdof] = dof
#     end
#   end
#   return mdof_to_dof
# end

# function compose_index(i1_to_i2,i2_to_i3)
#   T_i3 = eltype(i2_to_i3)
#   n_i2 = length(i1_to_i2)
#   i1_to_i3 = zeros(T_i3,n_i2)
#   for (i1,i2) in enumerate(i1_to_i2)
#     i1_to_i3[i1] = i2_to_i3[i2]
#   end
#   return i1_to_i3
# end

# mdof_to_dof = get_mdof_to_dof(Vagg)


# cell_dofs = get_cell_dof_ids(V)
# ndofs = maximum(cell_dofs.data)
# ptrs = zeros(Int32,ndofs+1)
# for dof in cell_dofs.data
#   ptrs[dof+1] += 1
# end
# length_to_ptrs!(ptrs)
# data = Vector{Int32}(undef,ptrs[end]-1)
# for cell in 1:length(cell_dofs)
#   pini = cell_dofs.ptrs[cell]
#   pend = cell_dofs.ptrs[cell+1]-1
#   for p in pini:pend
#     dof = cell_dofs.data[p]
#     data[ptrs[dof]] = cell
#     ptrs[dof] += 1
#   end
# end
# rewind_ptrs!(ptrs)
# yea = Table(data,ptrs)
