using Gridap
using Gridap.FESpaces
using GridapEmbedded
using ROManifolds
using ROManifolds.DofMaps
using ROManifolds.TProduct
using ROManifolds.Utils
using ROManifolds.ParamAlgebra
using ROManifolds.ParamSteady
using ROManifolds.RBSteady
using SparseArrays
using DrWatson
using Test

pdomain = (1,10,1,10,1,10)
pspace = ParamSpace(pdomain)

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

order = 2
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΩout = Measure(Ωactout,degree)
Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)
Γ = EmbeddedBoundary(cutgeo)
n_Γ = get_normal_vector(Γ)
dΓ = Measure(Γ,degree)

ν(μ) = x->μ[3]
νμ(μ) = ParamFunction(ν,μ)

f(μ) = x->μ[1]*x[1] - μ[2]*x[2]
fμ(μ) = ParamFunction(f,μ)

h(μ) = x->1
hμ(μ) = ParamFunction(h,μ)

g(μ) = x->μ[3]*x[1]-x[2]
gμ(μ) = ParamFunction(g,μ)

const γd = 10.0
const hd = dp[1]/n

a(μ,u,v,dΩ,dΓ) = ∫(νμ(μ)*∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
l(μ,v,dΩ,dΓ,dΓn) = ∫(fμ(μ)⋅v)dΩ + ∫(hμ(μ)⋅v)dΓn + ∫( (γd/hd)*v*gμ(μ) - (n_Γ⋅∇(v))*gμ(μ) )dΓ
res(μ,u,v,dΩ,dΓ,dΓn) =  a(μ,u,v,dΩ,dΓ) - l(μ,v,dΩ,dΓ,dΓn)

trian_a = (Ω,Γ)
trian_res = (Ω,Γ,Γn)
domains = FEDomains(trian_res,trian_a)

reffe = ReferenceFE(lagrangian,Float64,order)
Vact = FESpace(Ωact,reffe,conformity=:H1)
Vagg = AgFEMSpace(Vact,aggregates)
Uagg = ParamTrialFESpace(Vagg,gμ)

feop = LinearParamOperator(res,a,pspace,Uagg,Vagg,domains)

aout(u,v) = ∫(∇(v)⋅∇(u))dΩout
lout(μ,v) = ∫(∇(v)⋅∇(gμ(μ)))dΩout
Vout = FESpace(Ωactout,reffe,conformity=:H1)
Uout = ParamTrialFESpace(Vout,gμ)

V = FESpace(model,reffe,conformity=:H1)
U = ParamTrialFESpace(V,gμ)

μ = realization(feop;nparams=30)
ext = HarmonicExtension(V,Vagg,Uout,Vout,aout,lout,μ)
solver = ExtensionSolver(LUSolver(),ext)
nlop = parameterize(set_domains(feop),μ)
u = solve(solver,nlop)

struct NewDofMap{D,I<:AbstractVector} <: AbstractDofMap{D,Int32}
  size::Dims{D}
  indices::I
end

Base.size(i::NewDofMap) = i.size

Base.getindex(i::NewDofMap,j::Integer) = getindex(i.indices,j)

Base.setindex!(i::NewDofMap,v,j::Integer) = setindex!(i.indices,v,j)

function Base.reshape(i::NewDofMap,s::Vararg{Int})
  @assert prod(s) == length(i)
  NewDofMap(s,i.indices)
end

function DofMaps.flatten(i::NewDofMap)
  NewDofMap((prod(i.size),),i.indices)
end

function DofMaps.change_dof_map(i::NewDofMap,args...)
  NewDofMap(i,args...)
end

function DofMaps.change_dof_map(i::NewDofMap,i′::NewDofMap)
  i′
end

cell_odofs_ids = DofMaps.get_cell_odof_ids(V)
bg_dof_to_bg_odof = DofMaps.reorder_dofs(V,cell_odofs_ids)

s = (9,9)
dof_map = NewDofMap(s,bg_dof_to_bg_odof)

fesnaps = Snapshots(u,dof_map,μ)

## univariate stuff
tpmodel = TProductDiscreteModel(pmin,pmax,partition)
umodel = tpmodel.models_1d[1]
Ω1 = Triangulation(umodel)
dΩ1 = Measure(Ω1,degree)
V1 = FESpace(umodel,reffe,conformity=:H1)
V2 = FESpace(umodel,reffe,conformity=:H1)

cell_odofs_ids1 = DofMaps.get_cell_odof_ids(V1)
bg_dof_to_bg_odof1 = DofMaps.reorder_dofs(V1,cell_odofs_ids1)
##

A1 = assemble_matrix((u,v)->∫(∇(v)⋅∇(u))dΩ1,V1,V1)
A1′ = A1[bg_dof_to_bg_odof1,bg_dof_to_bg_odof1]
X1 = Rank1Tensor([A1′,A1′])

reduction = TTSVDReduction(fill(1e-4,2))
rbsolver = RBSolver(solver,reduction;nparams_res=30,nparams_jac=10)

red_style = reduction.red_style
Φ, = ttsvd(red_style,fesnaps,X1)
proj = TTSVDProjection(Φ,dof_map)

red_trial = reduced_subspace(Uagg,proj)
red_test = reduced_subspace(Vagg,proj)

##

A = assemble_matrix((u,v)->∫(∇(v)⋅∇(u))dΩbg,V,V)
A′ = A[bg_dof_to_bg_odof,bg_dof_to_bg_odof]
sparsity = TProductSparsity(SparsityPattern(A′),SparsityPattern.([A1′,A1′]))
sparse_dof_map = get_sparse_dof_map(sparsity,V,V)

Agg = assemble_matrix((u,v)->∫(∇(v)⋅∇(u))dΩ,Vagg,Vagg)
iagg,jagg, = findnz(Agg)

nV = num_free_dofs(V)

i_to_bg = similar(iagg)
for (k,ik) in enumerate(iagg)
  i_to_bg[k] = agg_dof_to_bg_odof[ik]
end

j_to_bg = similar(jagg)
for (k,jk) in enumerate(jagg)
  j_to_bg[k] = agg_dof_to_bg_odof[jk]
end

ij_to_bg = similar(i_to_bg)
for k in 1:length(iagg)
  ij_to_bg[k] = (j_to_bg[k]-1)*nV + i_to_bg[k]
end

jacs = jacobian_snapshots(rbsolver,feop,fesnaps)

iV,jV, = findnz(A′)

# jacs′ = contribution(jacs.trians) do trian
#   vals = jacs[trian]
#   ndata = maximum(sparse_dof_map)
#   pndata = ndata*nparams_jac
#   data = zeros(pndata)
#   for (k,bgk) in enumerate(ij_to_bg)
#     for (ip,bgkp) in enumerate(bgk:ndata:pndata)
#       data[bgkp] = vals[k,ip]
#     end
#   end
#   pdata = ConsecutiveParamArray(data)
#   spdata = sparse(nV,nV,j_to_bg,j_to_bg,pdata)
#   Snapshots(spdata,sparse_dof_map,μ)
# end
nparams_jac = 10
μjac = μ[1:nparams_jac]
jacs′ = contribution(jacs.trians) do trian
  vals = jacs[trian]
  ndata = maximum(sparse_dof_map)
  matvec = Vector{SparseMatrixCSC{Float64,Int}}(undef,nparams_jac)
  for ip in 1:nparams_jac
    data = zeros(nV,nV)
    for (k,(ibgk,jbgk)) in enumerate(zip(i_to_bg,j_to_bg))
      data[ibgk,jbgk] = vals[k,ip]
    end
    matvec[ip] = sparse(data)
  end
  spdata = ParamArray(matvec)
  Snapshots(spdata,sparse_dof_map,μjac)
end

ress = residual_snapshots(rbsolver,feop,fesnaps)
ext_ids = DofMaps.get_extension_dof_ids(ext)
agg_ids = setdiff(1:nV,ext_ids)
nparams_res = 30
μres = μ[1:nparams_res]
ress′ = contribution(ress.trians) do trian
  vals = ress[trian]
  data = zeros(nV,nparams_res)
  @views data[agg_ids,:] .= vals
  pdata = ConsecutiveParamArray(data)
  Snapshots(pdata,dof_map,μres)
end

lhs = reduced_jacobian(rbsolver.jacobian_reduction,red_trial,red_test,jacs′)
rhs = reduced_residual(rbsolver.residual_reduction,red_test,ress′)
trians_rhs = get_domains(rhs)
trians_lhs = get_domains(lhs)
feop′ = change_domains(feop,trians_rhs,trians_lhs)
rbop = GenericRBOperator(feop′,red_trial,red_test,lhs,rhs)

μon = realization(feop;sampling=:uniform,nparams=10)

# x̂,rbstats = solve(rbsolver,rbop,μon)
using Gridap.FESpaces
using ROManifolds.ParamDataStructures
_x = global_parameterize(zeros(size(u[1])),10)
_x̂ = project(get_trial(rbop)(μon),_x)
x̂ = RBParamVector(_x̂,_x)
nlop = parameterize(rbop,μon)
syscache = allocate_systemcache(nlop,x̂)
solve!(x̂,solver.solver,nlop,syscache)

inv_project(get_trial(rbop)(μon),x̂)

nlopon = parameterize(set_domains(feop),μon)
xon = solve(solver,nlopon)
fesnapson = Snapshots(xon,reshape(dof_map,nV),μon)

basis = get_basis(proj)
X = kron(A1′,A1′)
Uon = xon.data
Uon - basis*basis'*X*Uon

# no dof ordering

nV = num_free_dofs(V)
dof_map = VectorDofMap((nV,))
fesnaps = Snapshots(u,dof_map,μ)

reduction = Reduction(1e-4)
rbsolver = RBSolver(solver,reduction;nparams_res=30,nparams_jac=10)
X = assemble_matrix((u,v)->∫(∇(v)⋅∇(u))dΩbg,V,V)
red_style = reduction.red_style
Φ, = tpod(red_style,fesnaps,X)
proj = PODProjection(Φ)

# fesnaps - Φ*Φ'*X*fesnaps OK

red_trial = reduced_subspace(V,proj)
red_test = reduced_subspace(U,proj)

nparams_hr = 30
μhr = μ[1:nparams_hr]

ext_ids = DofMaps.get_extension_dof_ids(ext)
agg_ids = setdiff(1:nV,ext_ids)

# external vals
assem = SparseMatrixAssembler(Vout,Vout)
passem = parameterize(assem,μhr)
du = get_trial_fe_basis(Vout)
v = get_fe_basis(Vout)
matdata = collect_cell_matrix(Vout,Vout,aout(du,v))
vecdata = collect_cell_vector(Vout,lout(μ,v))
laplacian = assemble_matrix(assem,matdata)
residual = assemble_vector(passem,vecdata)

ress = residual_snapshots(rbsolver,feop,fesnaps)
ress′ = contribution(ress.trians) do trian
  vals = copy(ress[trian])
  data = zeros(nV,nparams_hr)
  @views data[agg_ids,:] .= vals
  # @views data[ext.dof_ids,:] .= residual.data[ext.ldof_ids,:]
  pdata = ConsecutiveParamArray(data)
  Snapshots(pdata,dof_map,μhr)
end

# ress′[1] - Φ*Φ'*X*ress′[1] OK

nparams_hj = 10
μhj = μ[1:nparams_hj]

Xagg = assemble_matrix((u,v)->∫(∇(v)⋅∇(u))dΩbg,Vagg,Vagg)
XaggΓ = assemble_matrix((u,v)->∫(u⋅v)dΓ,Vagg,Vagg)
agg_dof_to_bg_dof = DofMaps.get_dof_to_bg_dof(V,Vagg)

ibg,jbg, = findnz(X)
iagg,jagg, = findnz(Xagg)
iaggΓ,jaggΓ, = findnz(XaggΓ)

iagg_to_ibg = similar(iagg)
jagg_to_jbg = similar(jagg)
iagg_to_ibgΓ = similar(iaggΓ)
jagg_to_jbgΓ = similar(jaggΓ)

for k in eachindex(iagg_to_ibg)
  iagg_to_ibg[k] = agg_dof_to_bg_dof[iagg[k]]
  jagg_to_jbg[k] = agg_dof_to_bg_dof[jagg[k]]
end
for k in eachindex(iagg_to_ibgΓ)
  iagg_to_ibgΓ[k] = agg_dof_to_bg_dof[iaggΓ[k]]
  jagg_to_jbgΓ[k] = agg_dof_to_bg_dof[jaggΓ[k]]
end

sparsity′ = SparsityPattern(sparse(iagg_to_ibg,jagg_to_jbg,Xagg.nzval,nV,nV))
sparse_dof_map = TrivialSparseMatrixDofMap(sparsity′)

sparsityΓ′ = SparsityPattern(sparse(iagg_to_ibgΓ,jagg_to_jbgΓ,XaggΓ.nzval,nV,nV))
sparse_dof_mapΓ = TrivialSparseMatrixDofMap(sparsityΓ′)

sparse_dof_maps = Dict(Ω=>sparse_dof_map,Γ=>sparse_dof_mapΓ)

jacs = jacobian_snapshots(rbsolver,feop,fesnaps)
nzV = nnz(X)
jacs′ = contribution(jacs.trians) do trian
  vals = copy(jacs[trian].data.data)
  pdata = sparse(iagg_to_ibg,jagg_to_jbg,vals,nV,nV)
  sparse_dof_map = sparse_dof_maps[trian]
  Snapshots(pdata,sparse_dof_map,μhj)
end

lhs = reduced_jacobian(rbsolver.jacobian_reduction,red_trial,red_test,jacs′)
rhs = reduced_residual(rbsolver.residual_reduction,red_test,ress′)
trians_rhs = get_domains(rhs)
trians_lhs = get_domains(lhs)
feop′ = change_domains(feop,trians_rhs,trians_lhs)
rbop = GenericRBOperator(feop′,red_trial,red_test,lhs,rhs)

μon = realization(feop;sampling=:uniform,nparams=10)
# x̂,rbstats = solve(rbsolver,rbop,μon)
using ROManifolds.ParamDataStructures
_x = global_parameterize(zeros(size(u[1])),10)
_x̂ = project(get_trial(rbop)(μon),_x)
x̂ = RBParamVector(_x̂,_x)
nlop = parameterize(rbop,μon)
syscache = allocate_systemcache(nlop,x̂)
solve!(x̂,solver.solver,nlop,syscache)

nlopon = parameterize(set_domains(feop),μon)
x = solve(solver,nlopon)
x = Snapshots(x,dof_map,μon)

perf = eval_performance(rbsolver,feop,rbop,x,x̂,CostTracker(),CostTracker())
