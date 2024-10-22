using Gridap
using Gridap.Algebra
using Gridap.CellData
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.ODEs
using Test
using DrWatson
using Serialization
using BenchmarkTools

using ReducedOrderModels

# time marching
θ = 0.5
dt = 0.0025
t0 = 0.0
tf = 60*dt

# parametric space
pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

order = 2
degree = 2*order+1

const Re′ = 100.0
a(x,μ,t) = μ[1]/Re′
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

const W = 0.5
inflow(μ,t) = abs(1-cos(2π*t/tf)+μ[3]*sin(μ[2]*2π*t/tf)/100)
g_in(x,μ,t) = VectorValue(-x[2]*(W-x[2])*inflow(μ,t),0.0,0.0)
g_in(μ,t) = x->g_in(x,μ,t)
gμt_in(μ,t) = TransientParamFunction(g_in,μ,t)
g_0(x,μ,t) = VectorValue(0.0,0.0,0.0)
g_0(μ,t) = x->g_0(x,μ,t)
gμt_0(μ,t) = TransientParamFunction(g_0,μ,t)

u0(x,μ) = VectorValue(0.0,0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

# # loop on mesh
# for h in ("h007","h005","h0035")
#   model_dir = datadir(joinpath("models","model_circle_$h.json"))
#   model = DiscreteModelFromFile(model_dir)
#   labels = get_face_labeling(model)
#   add_tag_from_tags!(labels,"dirichlet_noslip",["cylinders","walls"])
#   add_tag_from_tags!(labels,"dirichlet_nopenetration",["top_bottom"])
#   add_tag_from_tags!(labels,"dirichlet",["inlet"])

#   Ω = Triangulation(model)
#   dΩ = Measure(Ω,degree)

#   c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
#   dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

#   djac(μ,t,(uₜ,pₜ),(dut,dpt),(v,q)) = ∫(v⋅dut)dΩ
#   jac(μ,t,(u,p),(du,dp),(v,q)) = dc(u,du,v) + ∫(aμt(μ,t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ - ∫(q*(∇⋅(du)))dΩ
#   res(μ,t,(u,p),(v,q)) = c(u,v) + ∫(v⋅∂t(u))dΩ + ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ

#   reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
#   test_u = TestFESpace(model,reffe_u;conformity=:H1,
#     dirichlet_tags=["inlet","dirichlet_nopenetration","dirichlet_noslip"],
#     dirichlet_masks=[[true,true,true],[false,false,true],[true,true,true]])
#   trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_in,gμt_0])
#   reffe_p = ReferenceFE(lagrangian,Float64,order-1)
#   test_p = TestFESpace(model,reffe_p;conformity=:C0)
#   trial_p = TrialFESpace(test_p)
#   test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
#   trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
#   feop = TransientParamFEOperator(res,(jac,djac),ptspace,trial,test)

#   xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

#   odeop = get_algebraic_operator(feop)
#   ws = (1,1)
#   us(x) = (x,x)

#   # loop on params
#   for nparams in 1:10
#     r = realization(ptspace;nparams)

#     U = trial(r)
#     x = get_free_dof_values(zero(U))

#     paramcache = allocate_paramcache(odeop,r,(x,x))
#     stageop = NonlinearParamStageOperator(odeop,paramcache,r,us,ws)

#     println("Residual time with h = $h, nparams = $nparams:")
#     @btime residual!(allocate_residual($stageop,$x),$stageop,$x)

#     println("Jacobian time with h = $h, nparams = $nparams:")
#     @btime jacobian!(allocate_jacobian($stageop,$x),$stageop,$x)

#     # println("Solve time with h = $h, nparams = $nparams:")
#     # @btime solve!(x,LUSolver(),A,b)
#   end
# end

# # loop on mesh
# μ = rand(3)
# g′_in(x,t) = VectorValue(-x[2]*(W-x[2])*inflow(μ,t),0.0,0.0)
# g′_in(t) = x->g′_in(x,t)
# g′_0(x,t) = VectorValue(0.0,0.0,0.0)
# g′_0(t) = x->g′_0(x,t)

# for h in ("h007","h005","h0035")
#   model_dir = datadir(joinpath("models","model_circle_$h.json"))
#   model = DiscreteModelFromFile(model_dir)
#   labels = get_face_labeling(model)
#   add_tag_from_tags!(labels,"dirichlet_noslip",["cylinders","walls"])
#   add_tag_from_tags!(labels,"dirichlet_nopenetration",["top_bottom"])
#   add_tag_from_tags!(labels,"dirichlet",["inlet"])

#   Ω = Triangulation(model)
#   dΩ = Measure(Ω,degree)

#   c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
#   dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

#   djac(t,(uₜ,pₜ),(dut,dpt),(v,q)) = ∫(v⋅dut)dΩ
#   jac(t,(u,p),(du,dp),(v,q)) = dc(u,du,v) + ∫(a(μ,t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ - ∫(q*(∇⋅(du)))dΩ
#   res(t,(u,p),(v,q)) = c(u,v) + ∫(v⋅u)dΩ + ∫(a(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ

#   reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
#   test_u = TestFESpace(model,reffe_u;conformity=:H1,
#     dirichlet_tags=["inlet","dirichlet_nopenetration","dirichlet_noslip"],
#     dirichlet_masks=[[true,true,true],[false,false,true],[true,true,true]])
#   trial_u = TransientTrialFESpace(test_u,[g′_in,g′_in,g′_0])
#   reffe_p = ReferenceFE(lagrangian,Float64,order-1)
#   test_p = TestFESpace(model,reffe_p;conformity=:C0)
#   trial_p = TrialFESpace(test_p)
#   test = TransientMultiFieldFESpace([test_u,test_p];style=BlockMultiFieldStyle())
#   trial = TransientMultiFieldFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
#   feop = TransientFEOperator(res,(jac,djac),trial,test)

#   xh0 = interpolate_everywhere([u0(μ),p0(μ)],trial(t0))

#   odeop = get_algebraic_operator(feop)
#   ws = (1,1)
#   us(x) = (x,x)

#   U = trial(t0)
#   x = get_free_dof_values(zero(U))

#   paramcache = allocate_paramcache(odeop,t0,(x,x))
#   stageop = NonlinearStageOperator(odeop,paramcache,t0,us,ws)

#   println("Residual time with h = $h:")
#   @btime residual!(allocate_residual($stageop,$x),$stageop,$x)

#   println("Jacobian time with h = $h:")
#   @btime jacobian!(allocate_jacobian($stageop,$x),$stageop,$x)

#   # println("Solve time with h = $h:")
#   # @btime solve!(x,LUSolver(),A,b)
# end


h ="h0035"
model_dir = datadir(joinpath("models","model_circle_$h.json"))
model = DiscreteModelFromFile(model_dir)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet_noslip",["cylinders","walls"])
add_tag_from_tags!(labels,"dirichlet_nopenetration",["top_bottom"])
add_tag_from_tags!(labels,"dirichlet",["inlet"])

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

djac(μ,t,(uₜ,pₜ),(dut,dpt),(v,q)) = ∫(v⋅dut)dΩ
jac(μ,t,(u,p),(du,dp),(v,q)) = dc(u,du,v) + ∫(aμt(μ,t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ - ∫(q*(∇⋅(du)))dΩ
res(μ,t,(u,p),(v,q)) = c(u,v) + ∫(v⋅∂t(u))dΩ + ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,
  dirichlet_tags=["inlet","dirichlet_nopenetration","dirichlet_noslip"],
  dirichlet_masks=[[true,true,true],[false,false,true],[true,true,true]])
trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_in,gμt_0])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = TransientParamFEOperator(res,(jac,djac),ptspace,trial,test)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

odeop = get_algebraic_operator(feop)
ws = (1,1)
us(x) = (x,x)

nparams = 1
r = realization(ptspace;nparams)

U = trial(r)
x = get_free_dof_values(zero(U))

paramcache = allocate_paramcache(odeop,r,(x,x))
stageop = NonlinearParamStageOperator(odeop,paramcache,r,us,ws)

# A = allocate_jacobian(stageop,x)
odeop,paramcache = stageop.odeop,stageop.paramcache
rx = stageop.rx
usx = stageop.usx(x)
# allocate_jacobian(odeop,rx,usx,paramcache)
uh = ODEs._make_uh_from_us(odeop,usx,paramcache.trial)
trial = evaluate(get_trial(odeop.op),nothing)
du = get_trial_fe_basis(trial)
test = get_test(odeop.op)
v = get_fe_basis(test)
assem = get_param_assembler(odeop.op,rx)

μ,t = get_params(rx),get_times(rx)

jacs = get_jacs(odeop.op)
Dc = DomainContribution()
for k in 1:2
  Jac = jacs[k]
  Dc = Dc + Jac(μ,t,uh,du,v)
end
matdata = collect_cell_matrix(trial,test,Dc)
A = allocate_matrix(assem,matdata)

# jacobian!(A,stageop,x)
# assemble_matrix_add!(A,assem,matdata)
using BlockArrays
using Gridap.Algebra
using Gridap.Arrays
using Gridap.Fields

m1 = ArrayBlock(blocks(A),fill(true,blocksize(A)))
m2 = MultiField.expand_blocks(assem,m1)
# FESpaces.assemble_matrix_add!(m2,assem,matdata)
# numeric_loop_matrix!(m2,assem,matdata)
(cellmat,cellidsrows,cellidscols) = first(zip(matdata...))
rows_cache = array_cache(cellidsrows)
cols_cache = array_cache(cellidscols)
vals_cache = array_cache(cellmat)
mat1 = getindex!(vals_cache,cellmat,1)
rows1 = getindex!(rows_cache,cellidsrows,1)
cols1 = getindex!(cols_cache,cellidscols,1)
add! = FESpaces.AddEntriesMap(+)
add_cache = return_cache(add!,A,mat1,rows1,cols1)

for cell in 1:length(cellidscols)
  rows = getindex!(rows_cache,cellidsrows,cell)
  cols = getindex!(cols_cache,cellidscols,cell)
  vals = getindex!(vals_cache,cellmat,cell)
  evaluate!(add_cache,add!,m2,vals,rows,cols)
end

cell = 1
# vals = getindex!(vals_cache,cellmat,cell)
cache = vals_cache
_cache, index_and_item = cache
index = LinearIndices(cellmat)[cell...]
# if index_and_item.index != index
cg, cgi, cf = _cache
gi = getindex!(cg, cellmat.maps, cell...)
index_and_item.item = Arrays._getindex_and_call!(cgi,gi,cf,cellmat.args,cell...)
index_and_item.index = index


#############

μ = rand(3)
g′_in(x,t) = VectorValue(-x[2]*(W-x[2])*inflow(μ,t),0.0,0.0)
g′_in(t) = x->g′_in(x,t)
g′_0(x,t) = VectorValue(0.0,0.0,0.0)
g′_0(t) = x->g′_0(x,t)

c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

djac(t,(uₜ,pₜ),(dut,dpt),(v,q)) = ∫(v⋅dut)dΩ
jac(t,(u,p),(du,dp),(v,q)) = dc(u,du,v) + ∫(a(μ,t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ - ∫(q*(∇⋅(du)))dΩ
res(t,(u,p),(v,q)) = c(u,v) + ∫(v⋅u)dΩ + ∫(a(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,
  dirichlet_tags=["inlet","dirichlet_nopenetration","dirichlet_noslip"],
  dirichlet_masks=[[true,true,true],[false,false,true],[true,true,true]])
trial_u = TransientTrialFESpace(test_u,[g′_in,g′_in,g′_0])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = TransientFEOperator(res,(jac,djac),trial,test)

xh0 = interpolate_everywhere([u0(μ),p0(μ)],trial(t0))

odeop = get_algebraic_operator(feop)
ws = (1,1)
us(x) = (x,x)

U = trial(t0)
x = get_free_dof_values(zero(U))

paramcache = allocate_paramcache(odeop,t0,(x,x))
stageop = NonlinearStageOperator(odeop,paramcache,t0,us,ws)

usx = stageop.usx(x)
A = allocate_jacobian(odeop,t0,usx,paramcache)

uh = ODEs._make_uh_from_us(odeop,usx,paramcache.trial)
trial = evaluate(trial,nothing)
du = get_trial_fe_basis(trial)
v = get_fe_basis(test)
assem = get_assembler(feop)

Dc = DomainContribution()
for k in 1:2
  Jac = feop.jacs[k]
  Dc = Dc + Jac(t,uh,du,v)
end
matdata = collect_cell_matrix(trial,test,Dc)

# jacobian!(A,stageop,x)
# assemble_matrix_add!(A,assem,matdata)

m1 = ArrayBlock(blocks(A),fill(true,blocksize(A)))
m2 = MultiField.expand_blocks(assem,m1)
# FESpaces.assemble_matrix_add!(m2,assem,matdata)
# numeric_loop_matrix!(m2,assem,matdata)
(cellmat,cellidsrows,cellidscols) = first(zip(matdata...))
rows_cache = array_cache(cellidsrows)
cols_cache = array_cache(cellidscols)
vals_cache = array_cache(cellmat)
mat1 = getindex!(vals_cache,cellmat,1)
rows1 = getindex!(rows_cache,cellidsrows,1)
cols1 = getindex!(cols_cache,cellidscols,1)
add! = FESpaces.AddEntriesMap(+)
add_cache = return_cache(add!,A,mat1,rows1,cols1)

for cell in 1:length(cellidscols)
  rows = getindex!(rows_cache,cellidsrows,cell)
  cols = getindex!(cols_cache,cellidscols,cell)
  vals = getindex!(vals_cache,cellmat,cell)
  evaluate!(add_cache,add!,m2,vals,rows,cols)
end

################################################################################

function mylength_to_ptrs!(ptrs)
  ptrs[1] = one(eltype(ptrs))
  n = length(ptrs)
  @inbounds for i in 1:(n-1)
      ptrs[i+1] += ptrs[i]
  end
  ptrs
end

struct MyNewParamMatrix{Tv,Ti} <: AbstractMatrix{Tv}
  data::Vector{Tv}
  ptrs::Vector{Ti}
  nrows::Vector{Ti}
end

function MyNewParamMatrix(
  a::AbstractVector{<:AbstractMatrix{Tv}},
  ptrs::Vector{Ti},
  nrows::Vector{Ti}) where {Tv,Ti}

  n = length(a)
  u = one(Ti)
  ndata = ptrs[end]-u
  data = Vector{Tv}(undef,ndata)
  p = 1
  @inbounds for i in 1:n
    ai = a[i]
    for j in 1:length(ai)
      aij = ai[j]
      data[p] = aij
      p += 1
    end
  end
  MyNewParamMatrix(data,ptrs,nrows)
end

function MyNewParamMatrix(a::AbstractVector{<:AbstractMatrix{Tv}}) where Tv
  Ti = Int
  n = length(a)
  ptrs = Vector{Ti}(undef,n+1)
  nrows = Vector{Ti}(undef,n)
  @inbounds for i in 1:n
    ai = a[i]
    ptrs[i+1] = length(ai)
    nrows[i] = size(ai,1)
  end
  mylength_to_ptrs!(ptrs)
  MyParamArray(a,ptrs,nrows)
end

Base.size(A::MyNewParamMatrix) = (length(A.ptrs)-1,length(A.ptrs)-1)

function Base.getindex(A::MyNewParamMatrix{Tv,Ti},i::Integer,j::Integer) where {Tv,Ti}
  @boundscheck checkbounds(A,i,j)
  u = one(eltype(A.ptrs))
  pini = A.ptrs[i]
  pend = A.ptrs[i+1]-u
  nrow = A.nrows[i]
  ncol = Int((pend-pini+1)/nrow)
  if i == j
    reshape(A.data[pini:pend],nrow,ncol)
  else
    fill(zero(Tv),nrow,ncol)
  end
end

function Base.setindex!(A::MyNewParamMatrix,v,i::Integer,j::Integer)
  @boundscheck checkbounds(A,i,j)
  u = one(eltype(A.ptrs))
  pini = A.ptrs[i]
  pend = A.ptrs[i+1]-u
  if i == j
    A.data[pini:pend] = vec(v)
  end
end

vec_of_mat = [rand(3,3),rand(4,4),rand(5,5)]

pmat = MyNewParamMatrix(vec_of_mat)
