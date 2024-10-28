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
using ReducedOrderModels.ParamDataStructures
using ReducedOrderModels.ParamSteady
using ReducedOrderModels.ParamFESpaces

# time marching
θ = 0.5
dt = 0.0025
t0 = 0.0
tf = 20*dt

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

# loop on mesh
for h in ("h0035",) #("h007","h005","h0035")
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

  # loop on params
  for nparams in 1:10
    r = realization(ptspace;nparams)

    U = trial(r)
    x = get_free_dof_values(zero(U))

    paramcache = allocate_paramcache(odeop,r,(x,x))
    stageop = ParamStageOperator(odeop,paramcache,r,us,ws)

    # println("Residual time with h = $h, nparams = $nparams:")
    # @btime residual!(allocate_residual($stageop,$x),$stageop,$x)

    println("Jacobian time with h = $h, nparams = $nparams:")
    @btime jacobian!(allocate_jacobian($stageop,$x),$stageop,$x)

    # println("Solve time with h = $h, nparams = $nparams:")
    # @btime solve!(x,LUSolver(),A,b)
  end
end

b = allocate_residual(stageop,x)
A = allocate_jacobian(stageop,x)
@btime b = allocate_residual($stageop,$x)
@btime A = allocate_jacobian($stageop,$x)
@btime residual!($b,$stageop,$x)
@btime jacobian!($A,$stageop,$x)

μ1 = rand(3)
g′_in(x,t) = VectorValue(-x[2]*(W-x[2])*inflow(μ1,t),0.0,0.0)
g′_in(t) = x->g′_in(x,t)
g′_0(x,t) = VectorValue(0.0,0.0,0.0)
g′_0(t) = x->g′_0(x,t)

djac′(t,(uₜ,pₜ),(dut,dpt),(v,q)) = ∫(v⋅dut)dΩ
jac′(t,(u,p),(du,dp),(v,q)) = dc(u,du,v) + ∫(a(μ1,t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ - ∫(q*(∇⋅(du)))dΩ
res′(t,(u,p),(v,q)) = c(u,v) + ∫(v⋅u)dΩ + ∫(a(μ1,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ

trial_u′ = TransientTrialFESpace(test_u,[g′_in,g′_in,g′_0])
test′ = test
trial′ = TransientMultiFieldFESpace([trial_u′,trial_p];style=BlockMultiFieldStyle())
feop′ = TransientFEOperator(res′,(jac′,djac′),trial′,test′)

xh0′ = interpolate_everywhere([u0(μ1),p0(μ1)],trial′(t0))

odeop′ = get_algebraic_operator(feop′)

U′ = trial′(t0)
x′ = get_free_dof_values(zero(U′))

odeopcache = allocate_odeopcache(odeop′,t0,(x′,x′))
stageop′ = NonlinearStageOperator(odeop′,odeopcache,t0,us,ws)

b′ = allocate_residual(stageop′,x′)
A′ = allocate_jacobian(stageop′,x′)
@btime allocate_residual($stageop′,$x′)
@btime allocate_jacobian($stageop′,$x′)
@btime residual!($b′,$stageop′,$x′)
@btime jacobian!($A′,$stageop′,$x′)

#alloc
965.35/30.98
64.77/6.80
#!
1018.01/18.73
145.76/8.96

(1018.01+965.35) / (30.98+18.73)
(64.77+145.76) / (6.80+8.96)

@btime jacobian!(allocate_jacobian($stageop,$x),$stageop,$x)
@btime residual!(allocate_residual($stageop,$x),$stageop,$x)

@btime jacobian!(allocate_jacobian($stageop′,$x′),$stageop′,$x′)
@btime residual!(allocate_residual($stageop′,$x′),$stageop′,$x′)

1940/49.70
210.53/15.76

################################################################################
using Gridap.FESpaces
using Gridap.CellData
using Gridap.Algebra
using Gridap.ODEs
using Gridap.Fields
using ReducedOrderModels.ParamDataStructures
using ReducedOrderModels.ParamSteady
using ReducedOrderModels.ParamFESpaces
using ReducedOrderModels.ParamAlgebra

red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
sjac = select_snapshots(fesnaps,1:20)
us_jac = (get_values(sjac),)
r_jac = get_realization(sjac)
odeop = get_algebraic_operator(feop.op_linear)
iA = get_matrix_index_map(odeop)

A = jacobian(fesolver,odeop,r_jac,us_jac)
A′ = jacobian(fesolver,odeop,r_jac,us_jac)

SA = Snapshots(A,iA,r_jac)
SA′ = Snapshots(A′,iA,r_jac);

op = odeop
r = r_jac
x = us_jac[1]
usx = (x,x)
odeop = op
paramcache = allocate_paramcache(odeop,r,(x,x))
uh = ODEs._make_uh_from_us(odeop,usx,paramcache.trial)
trial = evaluate(get_trial(odeop.op),nothing)
du = get_trial_fe_basis(trial)
test = get_test(odeop.op)
v = get_fe_basis(test)
assem = get_param_assembler(odeop.op,r)

μ,t = get_params(r),get_times(r)

jacs = get_jacs(odeop.op)
Dc = DomainContribution()
for k in 1:2
  Dc = Dc + jacs[k](μ,t,uh,du,v)
end
matdata = collect_cell_matrix(trial,test,Dc)

function ParamAlgebra.new_nz_allocation(a::ArrayBlock)
  array = map(ParamAlgebra.new_nz_allocation,a.array)
  return ArrayBlock(array,a.touched)
end

m1 = nz_counter(get_matrix_builder(assem),(get_rows(assem),get_cols(assem)));
symbolic_loop_matrix!(m1,assem,matdata);
m2 = ParamAlgebra.new_nz_allocation(m1);
numeric_loop_matrix!(m2,assem,matdata);
# m3 = create_from_nz(m2);

k = 1
ndata = length(m2[3].rowval)
pndata = length(m2[3].nzval)
vv = copy(m2[3].nzval)
j = 1
pini = Int(m2[3].colptr[j])
pend = pini + Int(m2[3].colnnz[j]) - 1
p = pini
itt = collect(p:ndata:pndata)
il,l = 1,itt[1]
vv[k + (il-1)*ndata] = vv[l]
il,l = 2,itt[2]
vv[k + (il-1)*ndata] = vv[l]
il,l = 3,itt[3]
vv[k + (il-1)*ndata] = vv[l]
il,l = 4,itt[4]
vv[k + (il-1)*ndata] = vv[l]
il,l = 5,itt[5]
vv[k + (il-1)*ndata] = vv[l]

# for j in 1:m2[3].ncols
#   pini = Int(m2[3].colptr[j])
#   pend = pini + Int(m2[3].colnnz[j]) - 1
#   for p in pini:pend
#     @inbounds for (il,l) in enumerate(p:ndata:pndata)
#       α = k + (il-1)*ndata
#       vv[α] = vv[l]
#     end
#     k += 1
#   end
# end

m1′ = nz_counter(get_matrix_builder(assem),(get_rows(assem),get_cols(assem)));
symbolic_loop_matrix!(m1′,assem,matdata);
m2′ = nz_allocation(m1′);
numeric_loop_matrix!(m2′,assem,matdata);
# m3′ = create_from_nz(m2′);

vv′ = copy(m2′[3].nzval)
vv′.data[k,1] = vv′.data[p,1]
vv′.data[k,2] = vv′.data[p,2]
vv′.data[k,3] = vv′.data[p,3]
vv′.data[k,4] = vv′.data[p,4]
vv′.data[k,5] = vv′.data[p,5]

# for j in 1:m2′[3].ncols
#   pini = Int(m2′[3].colptr[j])
#   pend = pini + Int(m2′[3].colnnz[j]) - 1
#   for p in pini:pend
#     @inbounds for l in param_eachindex(m2′[3])
#       vv′.data[k,l] = vv′.data[p,l]
#     end
#     # m2′[1].rowval[k] = m2′[1].rowval[p]
#     k += 1
#   end
# end

# reshape(vv,nnz,:) ≈ vv′.data

# pnnz = nnz*plength
# resize!(vv,pnnz)
# reshape(vv,nnz,plength) ≈ vv′.data[1:nnz,:]

for k in param_eachindex(m3)
  @assert param_getindex(m3,k) ≈ param_getindex(m3′,k) "Failed at $k"
end

mycheck(m2,m2′)

function mycheck(m2::ArrayBlock,m2′::ArrayBlock)
  map(mycheck,m2.array[1:3],m2′.array[1:3])
end

function mycheck(m2,m2′)
  k = 1
  ndata = length(m2.rowval)
  pndata = length(m2.nzval)
  plength = param_length(m2)
  for j in 1:m2.ncols
    pini = Int(m2.colptr[j])
    pend = pini + Int(m2.colnnz[j]) - 1
    for p in pini:pend
      @inbounds for (il,l) in enumerate(p:ndata:pndata)
        α = k + (il-1)*ndata
        vv[α] = vv[l]
      end
      m2.rowval[k] = m2.rowval[p]
      k += 1
    end
  end

  @inbounds for j in 1:m2′.ncols
    m2′.colptr[j+1] = m2′.colnnz[j]
  end
  length_to_ptrs!(m2′.colptr)
  nnz = m2.colptr[end]-1
  pnnz = nnz*plength
  resize!(vv,pnnz)
  # ret = ConsecutiveParamSparseMatrixCSC(a.nrows,a.ncols,a.colptr,a.rowval,data)

  vv′ = copy(m2′.nzval)
  k = 1
  for j in 1:m2′.ncols
    pini = Int(m2′.colptr[j])
    pend = pini + Int(m2′.colnnz[j]) - 1
    for p in pini:pend
      @inbounds for l in param_eachindex(m2′)
        vv′.data[k,l] = vv′.data[p,l]
      end
      m2′.rowval[k] = m2′.rowval[p]
      k += 1
    end
  end

  @inbounds for j in 1:m2′.ncols
    m2′.colptr[j+1] = m2′.colnnz[j]
  end
  length_to_ptrs!(m2′.colptr)
  nnz′ = m2′.colptr[end]-1

  # @assert m2.rowval == m2′.rowval
  @assert m2.colptr == m2′.colptr

  # ret′ = ConsecutiveParamSparseMatrixCSC(a.nrows,a.ncols,a.colptr,a.rowval,data)

  pnnz = nnz*plength
  @assert reshape(vv,nnz,plength) ≈ vv′.data[1:nnz,:]
end

using BlockArrays
for k in param_eachindex(m3)
  m3k=param_getindex(m3,k)
  m3k′=param_getindex(m3′,k)
  for i in 1:4
    b=blocks(m3k)[i]
    b′=blocks(m3k)[i]
    # @assert size(b)==size(b′)
    # @assert b.rowval==b′.rowval
    # @assert b.colptr==b′.colptr
    @assert b.nzval≈b′.nzval
  end
end

ndata = m2[1].colptr[end]-1
vv = copy(m2[1].nzval)
vv′ = copy(m2′[1].nzval)
M2 = m2[1]
k = 1
for j in 1:M2.ncols
  pini = Int(M2.colptr[j])
  pend = pini + Int(M2.colnnz[j]) - 1
  for p in pini:pend
    @inbounds for (il,l) in enumerate(p:ndata:pndata)
      α = k + (il-1)*ndata
      vv[α] = vv[l]
      vv′.data[k,il] = vv′.data[p,il]
      @assert vv′.data[k,il] == vv[α] "j = $j, p = $p, il = $il"
    end
    m2[1].rowval[k] = m2[1].rowval[p]
    m2′[1].rowval[k] = m2′[1].rowval[p]
    k += 1
  end
end
@inbounds for j in 1:m2[1].ncols
  m2[1].colptr[j+1] = m2[1].colnnz[j]
  m2′[1].colptr[j+1] = m2′[1].colnnz[j]
end
length_to_ptrs!(m2[1].colptr)
length_to_ptrs!(m2′[1].colptr)
nnz = m2[1].colptr[end]-1
pnnz = nnz*plength

ww = copy(vv)
resize!(m2[1].rowval,nnz)
resize!(m2′[1].rowval,nnz)

δ = Int(length(ww)/plength) - nnz
if δ > 0
  @inbounds for l in 1:plength
    Base._deleteat!(ww,l*nnz+1,δ)
  end
end
MM = vv′.data[1:nnz,:]

kk = 1
for _ in 1:pnnz

  k += 1
end

m,n = 5,4
A = repeat(1:m,n)
keep = 3
δ = m-keep
Base._deleteat!(A,keep+1,δ)
