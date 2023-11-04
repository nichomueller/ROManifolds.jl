begin
  root = pwd()
  include("$root/src/Utils/Utils.jl")
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")
end

struct TempPTFEOperator <: PTFEOperator{Nonlinear}
  res::Function
  jacs::Tuple{Vararg{Function}}
  assem::Assembler
  pspace::PSpace
  trials::Tuple{Vararg{Any}}
  test::FESpace
  order::Integer
  function TempPTFEOperator(
    res::Function,jac::Function,jac_t::Function,pspace,trial,test)
    assem = SparseMatrixAssembler(trial,test)
    new(res,(jac,jac_t),assem,pspace,(trial,∂ₚt(trial)),test,1)
  end
end

function Base.getindex(op::TempPTFEOperator,row,col)
  if isa(get_test(op),MultiFieldFESpace)
    trials_col = get_trial(op)[col]
    test_row = op.test[row]
    sf(q,idx) = single_field(op,q,idx)
    res(μ,t,u,dv) = op.res(μ,t,sf(u,col),sf(dv,row))
    jac(μ,t,u,du,dv) = op.jacs[1](μ,t,sf(u,col),sf(du,col),sf(dv,row))
    jac_t(μ,t,u,dut,dv) = op.jacs[2](μ,t,sf(u,col),sf(dut,col),sf(dv,row))
    return TempPTFEOperator(res,jac,jac_t,op.pspace,trials_col,test_row)
  else
    return op
  end
end

get_residual(op::TempPTFEOperator) = op.res
get_jacobian(op::TempPTFEOperator) = op.jacs

struct TempPODEOperator <: PODEOperator{Nonlinear}
  feop::TempPTFEOperator
end

function TransientFETools.get_algebraic_operator(feop::TempPTFEOperator)
  TempPODEOperator(feop)
end

struct TempPTAlgebraicOperator <: PTAlgebraicOperator{Nonlinear}
  odeop::PODEOperator
  μ
  tθ
  dtθ::Float
  u0::PTArray
  ode_cache
  vθ::PTArray
end

function get_ptoperator(
  odeop::TempPODEOperator,μ,tθ,dtθ::Float,u0::PTArray,ode_cache,vθ::PTArray)
  TempPTAlgebraicOperator(odeop,μ,tθ,dtθ,u0,ode_cache,vθ)
end

function solve_step!(
  uf::PTArray,
  solver::PThetaMethod,
  op::PODEOperator,
  μ::AbstractVector,
  u0::PTArray,
  t0::Real,
  cache)

  dt = solver.dt
  solver.θ == 0.0 ? dtθ = dt : dtθ = dt*solver.θ
  tθ = t0+dtθ

  if isnothing(cache)
    ode_cache = allocate_cache(op,μ,tθ)
    vθ = similar(u0)
    vθ .= 0.0
    nl_cache = nothing
  else
    ode_cache,vθ,nl_cache = cache
  end

  ode_cache = update_cache!(ode_cache,op,μ,tθ)

  nlop = TempPTAlgebraicOperator(op,μ,tθ,dtθ,u0,ode_cache,vθ)

  nl_cache = solve!(uf,solver.nls,nlop,nl_cache)

  if 0.0 < solver.θ < 1.0
    @. uf = uf*(1.0/solver.θ)-u0*((1-solver.θ)/solver.θ)
  end

  cache = (ode_cache,vθ,nl_cache)
  tf = t0+dt
  return (uf,tf,cache)
end

function residual!(
  b::PTArray,
  op::TempPTAlgebraicOperator,
  x::PTArray,
  args...)

  uF = x
  vθ = op.vθ
  @. vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(b))
  fill!(b,z)
  residual!(b,op.odeop,op.μ,op.tθ,(uF,vθ),op.ode_cache,args...)
end

function residual_for_trian!(
  b::PTArray,
  op::TempPTAlgebraicOperator,
  x::PTArray,
  args...)

  uF = x
  vθ = op.vθ
  z = zero(eltype(b))
  fill!(b,z)
  residual_for_trian!(b,op.odeop,op.μ,op.tθ,(uF,vθ),op.ode_cache,args...)
end

function residual_for_idx!(
  b::PTArray,
  op::TempPTAlgebraicOperator,
  x::PTArray,
  args...)

  uF = x
  vθ = op.vθ
  z = zero(eltype(b))
  fill!(b,z)
  residual!(b,op.odeop,op.μ,op.tθ,(uF,vθ),op.ode_cache,args...)
end

function jacobian!(
  A::PTArray,
  op::TempPTAlgebraicOperator,
  x::PTArray,
  args...)

  uF = x
  vθ = op.vθ
  @. vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(A))
  fillstored!(A,z)
  jacobians!(A,op.odeop,op.μ,op.tθ,(uF,vθ),(1.0,1/op.dtθ),op.ode_cache,args...)
end

function jacobian!(
  A::PTArray,
  op::TempPTAlgebraicOperator,
  x::PTArray,
  i::Int,
  args...)

  uF = x
  vθ = op.vθ
  @. vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(A))
  fillstored!(A,z)
  γ = (1.0,1/op.dtθ)
  jacobian!(A,op.odeop,op.μ,op.tθ,(uF,vθ),i,γ[i],op.ode_cache,args...)
end

function jacobian_for_trian!(
  A::PTArray,
  op::TempPTAlgebraicOperator,
  x::PTArray,
  i::Int,
  args...)

  uF = x
  vθ = op.vθ
  z = zero(eltype(A))
  fillstored!(A,z)
  γ = (1.0,1/op.dtθ)
  jacobian_for_trian!(A,op.odeop,op.μ,op.tθ,(uF,vθ),i,γ[i],op.ode_cache,args...)
end

function jacobian_for_idx!(
  A::PTArray,
  op::TempPTAlgebraicOperator,
  x::PTArray,
  i::Int,
  args...)

  uF = x
  vθ = op.vθ
  z = zero(eltype(A))
  fillstored!(A,z)
  γ = (1.0,1/op.dtθ)
  jacobian!(A,op.odeop,op.μ,op.tθ,(uF,vθ),i,γ[i],op.ode_cache,args...)
end

function collect_compress_rhs_lhs(
  info::RBInfo,
  feop::TempPTFEOperator,
  fesolver::PThetaMethod,
  rbspace,
  snaps::DirBlockSnapshots,
  μ::Table)

  nsnaps_mdeim = info.nsnaps_mdeim
  θ = fesolver.θ

  snapsθ = recenter(snaps,fesolver.uh0(μ);θ)
  _snapsθ,_μ = snapsθ[1:nsnaps_mdeim],μ[1:nsnaps_mdeim]
  op = get_ptoperator(fesolver,feop,_snapsθ,_μ)
  rhs = collect_compress_rhs(info,op,rbspace)
  lhs = collect_compress_lhs(info,op,rbspace;θ)
  show(rhs),show(lhs)

  rhs,lhs
end

Base.:(∘)(::Function,::Tuple{Vararg{Union{Nothing,CellField}}}) = nothing

# struct DirBlockSnapshots{T} <: RBBlock{T,1}
#   snaps::BlockSnapshots{T}
#   dsnaps::Snapshots{T}

#   function DirBlockSnapshots(
#     v::Vector{<:Vector{<:PTArray{T}}},
#     vd::Vector{<:PTArray{T}}) where T

#     snaps = BlockSnapshots(v)
#     dsnaps = Snapshots(vd)
#     new{T}(snaps,dsnaps)
#   end
# end

# function collect_multi_field_solutions(
#   fesolver::PODESolver,
#   feop::TempPTFEOperator,
#   μ::Table)

#   uh0,t0,tf = fesolver.uh0,fesolver.t0,fesolver.tf
#   ode_op = get_algebraic_operator(feop)
#   u0 = get_free_dof_values(uh0(μ))
#   times = get_times(fesolver)
#   time_ndofs = get_time_ndofs(fesolver)
#   nparams = length(μ)
#   T = get_vector_type(feop.test)
#   uμt = PODESolution(fesolver,ode_op,μ,u0,t0,tf)
#   sols = Vector{Vector{PTArray{T}}}(undef,time_ndofs)
#   dir_sols = get_trial(feop)(μ,times).dirichlet_values
#   println("Computing fe solution: time marching across $time_ndofs instants, for $nparams parameters")
#   stats = @timed for (sol,n) in uμt
#     sols[n] = split_fields(feop.test,copy(sol))
#   end
#   println("Time marching complete")
#   return DirBlockSnapshots(sols,dir_sols),ComputationInfo(stats,nparams)
# end

# function Base.getindex(ds::DirBlockSnapshots,idx::Int)
#   if idx == 1
#     s = ds.dsnaps[idx]
#     ds = s.dsnaps[idx]
#     return vcat(vcat(s...),vcat(ds...))
#   else
#     return ds.dsnaps[idx]
#   end
# end

# function Base.getindex(s::DirBlockSnapshots,idx::UnitRange{Int})
#   nblocks = get_nblocks(s)
#   map(1:nblocks) do row
#     srow = s[row]
#     srow[idx]
#   end
# end

# function recenter(s::DirBlockSnapshots,uh0::PTFEFunction;θ::Real=1)
#   if θ == 1
#     return s
#   end
#   @notimplemented
# end

# nearest_neighbor(sols::DirBlockSnapshots,args...) = nearest_neighbor(sols.s,args...)

# struct DirBlockRBSpace{T} <: RBBlock{T,1}
#   blocks::Vector{RBSpace{T}}
#   drbspace::RBSpace{T}

#   function BlockRBSpace(blocks::Vector{RBSpace{T}}) where T
#     new{T}(blocks)
#   end

#   function BlockRBSpace(bases_space::Vector{Matrix{T}},bases_time::Vector{Matrix{T}}) where T
#     blocks = map(RBSpace,bases_space,bases_time)
#     BlockRBSpace(blocks)
#   end
# end

# function Base.show(io::IO,rb::BlockRBSpace)
#   for (row,block) in enumerate(rb)
#     nbs = size(block.basis_space,2)
#     nbt = size(block.basis_time,2)
#     print(io,"\n")
#     printstyled("RB SPACE INFO FIELD $row\n";underline=true)
#     print(io,"Reduced basis space with #(basis space, basis time) = ($nbs,$nbt)\n")
#   end
# end

# function field_offsets(rb::BlockRBSpace)
#   nblocks = get_nblocks(rb)
#   offsets = zeros(Int,nblocks+1)
#   @inbounds for block = 1:nblocks
#     ndofs = get_rb_ndofs(rb[block])
#     offsets[block+1] = offsets[block] + ndofs
#   end
#   offsets
# end

begin
  mesh = "cube2x2.json"
  bnd_info = Dict("dirichlet" => [1,2,3,4,5,7,8],"neumann" => [6])
  test_path = "$root/tests/navier-stokes/unsteady/$mesh"
  order = 2
  degree = 4

  model = get_discrete_model(test_path,mesh,bnd_info)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  ranges = fill([1.,10.],3)
  sampling = UniformSampling()
  pspace = PSpace(ranges,sampling)

  a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
  a(μ,t) = x->a(x,μ,t)
  aμt(μ,t) = PTFunction(a,μ,t)

  g(x,μ,t) = VectorValue(μ[1]*exp(-x[2]/μ[2])*abs(sin(μ[3]*t)),0)
  g(μ,t) = x->g(x,μ,t)

  u0(x,μ) = VectorValue(0,0)
  u0(μ) = x->u0(x,μ)
  u0μ(μ) = PFunction(u0,μ)
  p0(x,μ) = 0
  p0(μ) = x->p0(x,μ)
  p0μ(μ) = PFunction(p0,μ)

  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  c(u,v) = ∫ₚ(v⊙(conv∘(u,∇(u))),dΩ)
  dc(u,du,v) = ∫ₚ(v⊙(dconv∘(du,∇(du),u,∇(u))),dΩ)

  res(μ,t,(u,p),(v,q)) = ∫ₚ(v⋅∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⊙∇(u),dΩ) + c(u,v) - ∫ₚ(p*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(u)),dΩ)
  jac(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(aμt(μ,t)*∇(v)⊙∇(du),dΩ) + dc(u,du,v) - ∫ₚ(dp*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(du)),dΩ)
  jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = ∫ₚ(v⋅dut,dΩ)

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float},order)
  reffe_p = ReferenceFE(lagrangian,Float,order-1)
  test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial_u = PTTrialFESpace(test_u,g)
  test_p = TestFESpace(model,reffe_p;conformity=:H1,constraint=:zeromean)
  trial_p = TrialFESpace(test_p)
  test = PTMultiFieldFESpace([test_u,test_p])
  trial = PTMultiFieldFESpace([trial_u,trial_p])
  feop = TempPTFEOperator(res,jac,jac_t,pspace,trial,test)

  t0,tf,dt,θ = 0.,0.05,0.005,1
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial_u(μ,t0))
  ph0μ(μ) = interpolate_everywhere(p0μ(μ),trial_p(μ,t0))
  xh0μ(μ) = interpolate_everywhere([uh0μ(μ),ph0μ(μ)],trial(μ,t0))

  nls = NewtonRaphsonSolver(LUSolver(),1e-10,20)
  fesolver = PThetaMethod(nls,xh0μ,θ,dt,t0,tf)

  ϵ = 1e-4
  load_solutions = true
  save_solutions = true
  load_structures = false
  save_structures = true
  norm_style = [:l2,:l2]
  compute_supremizers = true
  nsnaps_state = 50
  nsnaps_mdeim = 30
  nsnaps_test = 10
  st_mdeim = false
  postprocess = true
  info = RBInfo(test_path;ϵ,norm_style,compute_supremizers,nsnaps_state,
    nsnaps_mdeim,nsnaps_test,st_mdeim,postprocess)
end

function temp_collector_res(op::TempPTAlgebraicOperator,rbspace::RBSpace)
  b = allocate_residual(op,op.u0)
  bs = get_basis_space(rbspace)
  array = map(collect(eachcol(bs))) do bsi
    _b = copy(b)
    x = PTArray([vcat(copy(bsi),zeros(Np)) for _ = 1:length(op.μ)*length(op.tθ)])
    residual!(_b,op,x)
    copy(_b)
  end
  NnzMatrix(hcat(array...))
end

function temp_contribution(
  info::RBInfo,
  op::PTAlgebraicOperator,
  nzm::NnzMatrix,
  trian::Triangulation,
  args...;
  kwargs...)

  basis_space = tpod(nzm;ϵ=info.ϵ)
  proj_bs = project_space(basis_space,args...;kwargs...)
  proj_bt = [rand(ntimes,1),rbspace[1].basis_time] # careful here
  interp_idx_space = get_interpolation_idx(basis_space)
  entire_interp_idx_space = recast_idx(nzm,interp_idx_space)
  entire_interp_idx_rows,_ = vec_to_mat_idx(entire_interp_idx_space,nzm.nrows)

  interp_bs = basis_space[interp_idx_space,:]
  lu_interp = lu(interp_bs)
  cell_dof_ids = get_cell_dof_ids(op.odeop.feop.test,trian)
  red_integr_cells = find_cells(entire_interp_idx_rows,cell_dof_ids)
  red_trian = view(trian,red_integr_cells)
  red_meas = get_measure(op.odeop.feop,red_trian)
  red_times = op.tθ
  integr_domain = RBIntegrationDomain(red_meas,red_times,entire_interp_idx_space)

  RBAffineDecomposition(proj_bs,proj_bt,lu_interp,integr_domain)
end

res_nl(μ,t,(u,p),(v,q)) = c(u,v)
jac_nl(μ,t,(u,p),(du,dp),(v,q)) = dc(u,du,v)
feop_nl = TempPTFEOperator(res_nl,jac_nl,jac_t,pspace,trial,test)

sols,params = load(info,(BlockSnapshots,Table))
rbspace = load(info,BlockRBSpace)
snapsθ = recenter(sols,fesolver.uh0(params);θ)
_snapsθ,_μ = snapsθ[1:nsnaps_mdeim],params[1:nsnaps_mdeim]

op_nl = get_ptoperator(fesolver,feop_nl,_snapsθ,_μ)
trian = Ω
Nu = length(get_free_dof_ids(test_u))
Np = length(get_free_dof_ids(test_p))
ress = temp_collector_res(op_nl[1],rbspace[1])
ad_c = temp_contribution(info,op_nl[1],ress,trian,rbspace[1])

times = get_times(fesolver)
ntimes = length(times)
snaps_test,params_test = sols[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
xn,μn = [PTArray(snaps_test[1][1:ntimes]),PTArray(snaps_test[2][1:ntimes])],params_test[1:1]

x = nearest_neighbor(sols,params,μn)
op_nl_online = get_ptoperator(fesolver,feop_nl,x,μn)
rhs_cache,lhs_cache = allocate_cache(op_nl_online,x)
coeff_cache,rb_cache = rhs_cache
coeff = rhs_coefficient!(coeff_cache,op_nl_online,ad_c;st_mdeim)
rb_res_nl = rb_contribution!(rb_cache,RBVecContributionMap(Float),ad_c,coeff)

c_ok(u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
dc_ok(u,du,v) = ∫(v⊙(dconv∘(du,∇(du),u,∇(u))))dΩ

jac_t_ok(t,(u,p),(dut,dpt),(v,q)) = nothing
jac_ok(t,(u,p),(du,dp),(v,q)) = dc_ok(u,du,v)
res_ok(t,(u,p),(v,q)) = c_ok(u,v)
g_ok(x,t) = g(x,μn[1],t)
g_ok(t) = x->g_ok(x,t)
trial_u_ok = TransientTrialFESpace(test_u,g_ok)
trial_ok = TransientMultiFieldFESpace([trial_u_ok,trial_p])
feop_ok = TransientFEOperator(res_ok,jac_ok,jac_t_ok,trial_ok,test)
ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
ode_cache_ok = allocate_cache(ode_op_ok)

bok = zeros(Nu)
C = []
for (kt,t) in enumerate(times)
  xk = x[kt]
  ode_cache_ok = Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  z = zero(eltype(bok))
  fill!(bok,z)
  residual!(bok,ode_op_ok,t,(xk,zeros(size(xk))),ode_cache_ok)
  push!(C,bok)
end
Cred = space_time_projection(NnzMatrix(C...),rbspace[1])

rb_res_nl[1] - Cred
