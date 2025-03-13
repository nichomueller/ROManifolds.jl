using Gridap
using GridapEmbedded
using ROManifolds
using ROManifolds.DofMaps
using ROManifolds.TProduct
using DrWatson
using Test

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

op = AffineFEOperator(a,l,Uagg,Vagg)

Vout = FESpace(Ωactout,reffe,conformity=:H1)
Uout = TrialFESpace(Vout,g)

V = FESpace(model,reffe,conformity=:H1)

agg_dof_to_bgdof =  get_dof_to_bg_dof(V,Vagg)
act_out_dof_to_bgdof = get_dof_to_bg_dof(V,Vout)
agg_out_dof_to_bgdof = setdiff(act_out_dof_to_bgdof,agg_dof_to_bgdof)

cell_odofs_ids = DofMaps.get_cell_odof_ids(V)

function reorder_dofs(f::FESpace,cell_odofs_ids::AbstractVector)
  odof_to_dof = zeros(Int32,num_free_dofs(f))
  cell_dofs_ids = get_cell_dof_ids(f)
  cache = array_cache(cell_dofs_ids)
  ocache = array_cache(cell_odofs_ids)
  for cell in 1:length(cell_dofs_ids)
    dofs = getindex!(cache,cell_dofs_ids,cell)
    odofs = getindex!(ocache,cell_odofs_ids,cell)
    iodof_to_idof = odofs.terms
    for iodof in eachindex(odofs)
      idof = iodof_to_idof[iodof]
      dof = dofs[idof]
      odof = odofs[iodof]
      if odof > 0
        odof_to_dof[odof] = dof
      end
    end
  end
  return odof_to_dof
end

function invert_dof_map(dof2_to_dof1::AbstractVector,dofs2::AbstractVector)
  dofs1 = similar(dofs2)
  for (i,dof2) in enumerate(dofs2)
    dofs1[i] = dof2_to_dof1[dof2]
  end
  return dofs1
end

# using Gridap.FESpaces
# function get_bg_dof_to_odof(
#   bg_f::SingleFieldFESpace,
#   f::SingleFieldFESpace,
#   dof_to_bg_odof::AbstractVector
#   )

#   bg_dof_to_all_odof = DofMaps.get_bg_odof_to_odof(bg_f,f)
#   bg_dof_to_odof = similar(dof_to_bg_odof)
#   for (i,bg_odof) in enumerate(dof_to_bg_odof)
#     bg_dof_to_odof[i] = bg_dof_to_all_odof[bg_odof]
#   end
#   return bg_dof_to_odof
# end

using Gridap.Arrays
function OrderedFEFunction(f::FESpace,fv)
  dv = zero_dirichlet_values(f)
  cell_dof_ids = get_cell_odof_ids(f)
  cell_values = lazy_map(Broadcasting(PosNegReindex(fv,dv)),cell_dof_ids)
  cell_ovalues = DofMaps.cell_ovalue_to_value(f,cell_values)
  cell_field = CellField(f,cell_ovalues)
  SingleFieldFEFunction(cell_field,cell_ovalues,fv,dv,f)
end

dofs_to_odofs = reorder_dofs(V,cell_odofs_ids)
agg_dof_to_bg_odof = invert_dof_map(dofs_to_odofs,agg_dof_to_bgdof)
agg_out_dof_to_bg_odof = invert_dof_map(dofs_to_odofs,agg_out_dof_to_bgdof)

op = AffineFEOperator(a,l,Uagg,Vagg)

# extend by g
gV = interpolate_everywhere(g,V)
gV_out = view(gV.free_values,agg_out_dof_to_bg_odof)
ext = FunctionalExtension(gV_out,agg_out_dof_to_bg_odof)
solver = ExtensionSolver(LUSolver(),ext)
u = solve(solver,op.op)
uh = FEFunction(V,u)
writevtk(Ωbg,datadir("sol_g.vtu"),cellfields=["uh"=>uh])

# agg_out_dof_to_act_odof = get_bg_dof_to_odof(V,Vout,agg_out_dof_to_bg_odof)
# boh = get_bg_dof_to_dof(V,Vout,agg_out_dof_to_bg_odof)

# # harmonic extension
# aout(u,v) = ∫(∇(v)⋅∇(u))dΩout
# lout(v) =  ∫(∇(v)⋅∇(g))dΩout

# A = assemble_matrix(aout,Uout,Vout)
# b = assemble_vector(lout,Vout)
# ext = HarmonicExtension(A,b,agg_out_dof_to_bg_odof,agg_out_dof_to_act_odof)
# solver = ExtensionSolver(LUSolver(),ext)

# u = solve(solver,op.op)
# uh = FEFunction(V,u)
# writevtk(Ωbg,datadir("sol_harm.vtu"),cellfields=["uh"=>uh])
