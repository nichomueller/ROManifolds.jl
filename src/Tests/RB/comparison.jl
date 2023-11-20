begin
  root = pwd()
  include("$root/src/Utils/Utils.jl")
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")
end

abstract type _RBAlgebraicContribution{T,N} end

struct _RBVecAlgebraicContribution{T} <: _RBAlgebraicContribution{T,1}
  dict::IdDict{Triangulation,RBVecAffineDecomposition{T}}
  function _RBVecAlgebraicContribution(::Type{T}) where T
    new{T}(IdDict{Triangulation,RBVecAffineDecomposition{T}}())
  end
end

struct _RBMatAlgebraicContribution{T} <: _RBAlgebraicContribution{T,2}
  dict::IdDict{Triangulation,RBMatAffineDecomposition{T}}
  function _RBMatAlgebraicContribution(::Type{T}) where T
    new{T}(IdDict{Triangulation,RBMatAffineDecomposition{T}}())
  end
end

CellData.num_domains(a::_RBAlgebraicContribution) = length(a.dict)
CellData.get_domains(a::_RBAlgebraicContribution) = keys(a.dict)
Base.iszero(a::_RBAlgebraicContribution) = num_domains(a) == 0

function CellData.get_contribution(
  a::_RBAlgebraicContribution,
  trian::Triangulation)

  if haskey(a.dict,trian)
    return a.dict[trian]
  else
    @unreachable """\n
    There is not contribution associated with the given mesh in this _RBAlgebraicContribution object.
    """
  end
end

Base.getindex(a::_RBAlgebraicContribution,trian::Triangulation) = get_contribution(a,trian)
Base.eltype(::_RBAlgebraicContribution{T,N} where N) where T = T

function compress_component!(
  contrib::_RBAlgebraicContribution,
  rbinfo::RBInfo,
  op::PTOperator,
  snaps::Vector{<:NnzMatrix},
  trian::Base.KeySet{Triangulation},
  args...;
  kwargs...)

  for (i,ti) in enumerate(trian)
    si = snaps[i]
    if !iszero(si)
      ci = RBAffineDecomposition(rbinfo,op,si,ti,args...;kwargs...)
      add_contribution!(contrib,ti,ci)
    end
  end
end

function CellData.add_contribution!(
  a::_RBAlgebraicContribution,
  trian::Triangulation,
  b)

  @check !haskey(a.dict,trian)
  a.dict[trian] = b
  a
end

function get_rb_ndofs(a::_RBAlgebraicContribution)
  trian = first([get_domains(a)...])
  get_rb_ndofs(a[trian])
end

function old_collect_compress_rhs_lhs(
  rbinfo::RBInfo,
  feop::PTFEOperator{Affine},
  fesolver::PThetaMethod,
  rbspace,
  params::Table)

  θ = fesolver.θ
  nsnaps_mdeim = rbinfo.nsnaps_mdeim
  μ = params[1:nsnaps_mdeim]
  op = get_ptoperator(fesolver,feop,rbspace,μ)

  println("Computing RB affine decomposition (linear)")
  rhs = old_collect_compress_rhs(rbinfo,op,rbspace)
  lhs = old_collect_compress_lhs(rbinfo,op,rbspace;θ)

  return rhs,lhs
end

function old_collect_compress_rhs(
  rbinfo::RBInfo,
  op::PTOperator,
  rbspace::RBSpace{T}) where T

  ress,trian = collect_residuals_for_trian(op)
  ad_res = _RBVecAlgebraicContribution(T)
  compress_component!(ad_res,rbinfo,op,ress,trian,rbspace)
  return ad_res
end

function old_collect_compress_lhs(
  rbinfo::RBInfo,
  op::PTOperator,
  rbspace::RBSpace{T};
  θ::Real=1) where T

  njacs = length(op.odeop.feop.jacs)
  ad_jacs = Vector{_RBMatAlgebraicContribution{T}}(undef,njacs)
  for i = 1:njacs
    combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*x-θ*y
    jacs,trian = collect_jacobians_for_trian(op;i)
    ad_jac_i = _RBMatAlgebraicContribution(T)
    compress_component!(ad_jac_i,rbinfo,op,jacs,trian,rbspace,rbspace;combine_projections)
    ad_jacs[i] = ad_jac_i
  end
  return ad_jacs
end

function collect_rhs_contributions!(
  cache,
  rbinfo::RBInfo,
  op::PTOperator,
  rbres::_RBVecAlgebraicContribution{T},
  rbspace::RBSpace{T}) where T

  coeff_cache,rb_cache = cache
  st_mdeim = rbinfo.st_mdeim
  k = RBVecContributionMap()
  rb_res_contribs = Vector{PTArray{Vector{T}}}(undef,num_domains(rbres))
  if iszero(rbres)
    nrow = get_rb_ndofs(rbspace)
    contrib = AffinePTArray(zeros(T,nrow),length(op.μ))
    rb_res_contribs[i] = contrib
  else
    for (i,t) in enumerate(get_domains(rbres))
      rbrest = rbres[t]
      coeff = rhs_coefficient!(coeff_cache,op,rbrest;st_mdeim)
      rb_res_contribs[i] = rb_contribution!(rb_cache,k,rbrest,coeff)
    end
  end
  return sum(rb_res_contribs)
end

function collect_lhs_contributions!(
  cache,
  rbinfo::RBInfo,
  op::PTOperator,
  rbjacs::Vector{_RBMatAlgebraicContribution{T}},
  rbspace::RBSpace{T}) where T

  njacs = length(rbjacs)
  rb_jacs_contribs = Vector{PTArray{Matrix{T}}}(undef,njacs)
  for i = 1:njacs
    rb_jac_i = rbjacs[i]
    rb_jacs_contribs[i] = collect_lhs_contributions!(cache,rbinfo,op,rb_jac_i,rbspace,rbspace;i)
  end
  return rb_jacs_contribs
end

function collect_lhs_contributions!(
  cache,
  rbinfo::RBInfo,
  op::PTOperator,
  rbjac::_RBMatAlgebraicContribution{T},
  rbspace_row::RBSpace{T},
  rbspace_col::RBSpace{T};
  kwargs...) where T

  coeff_cache,rb_cache = cache
  trian = get_domains(rbjac)
  st_mdeim = rbinfo.st_mdeim
  k = RBMatContributionMap()
  rb_jac_contribs = Vector{PTArray{Matrix{T}}}(undef,num_domains(rbjac))
  if iszero(rbjac)
    nrow = get_rb_ndofs(rbspace_row)
    ncol = get_rb_ndofs(rbspace_col)
    contrib = AffinePTArray(zeros(T,nrow,ncol),length(op.μ))
    rb_jac_contribs[i] = contrib
  else
    for (i,t) in enumerate(trian)
      rbjact = rbjac[t]
      coeff = lhs_coefficient!(coeff_cache,op,rbjact;st_mdeim,kwargs...)
      rb_jac_contribs[i] = rb_contribution!(rb_cache,k,rbjact,coeff)
    end
  end
  return sum(rb_jac_contribs)
end

function rhs_coefficient!(
  cache,
  op::PTOperator,
  rbres::RBVecAffineDecomposition;
  kwargs...)

  rcache,scache... = cache
  red_integr_res = assemble_rhs!(rcache,op,rbres)
  old_mdeim_solve!(scache,rbres,red_integr_res;kwargs...)
end

function assemble_rhs!(
  cache,
  op::PTOperator,
  rbres::RBVecAffineDecomposition)

  red_idx = rbres.integration_domain.idx
  red_times = rbres.integration_domain.times
  red_meas = rbres.integration_domain.meas

  cache = get_cache_at_times(cache,op.tθ,red_times)
  sols = get_solutions_at_times(op.u0,op.tθ,red_times)

  collect_residuals_for_idx!(cache,op,sols,red_idx,red_meas)
end

function lhs_coefficient!(
  cache,
  op::PTOperator,
  rbjac::RBMatAffineDecomposition;
  i::Int=1,kwargs...)

  jcache,scache... = cache
  red_integr_jac = assemble_lhs!(jcache,op,rbjac;i)
  old_mdeim_solve!(scache,rbjac,red_integr_jac;kwargs...)
end

function assemble_lhs!(
  cache,
  op::PTOperator,
  rbjac::RBMatAffineDecomposition;
  i::Int=1)

  red_idx = rbjac.integration_domain.idx
  red_times = rbjac.integration_domain.times
  red_meas = rbjac.integration_domain.meas

  cache = get_cache_at_times(cache,op.tθ,red_times)
  sols = get_solutions_at_times(op.u0,op.tθ,red_times)

  collect_jacobians_for_idx!(cache,op,sols,red_idx,red_meas;i)
end

function get_cache_at_times(q::NonaffinePTArray,times::Vector{<:Real},red_times::Vector{<:Real})
  time_ndofs = length(times)
  time_ndofs_red = length(red_times)
  nparams = Int(length(q)/time_ndofs)
  if length(red_times) < time_ndofs
    return NonaffinePTArray(q[1:time_ndofs_red*nparams])
  else
    return q
  end
end

function get_solutions_at_times(sols::NonaffinePTArray,times::Vector{<:Real},red_times::Vector{<:Real})
  time_ndofs = length(times)
  nparams = Int(length(sols)/time_ndofs)
  if length(red_times) < time_ndofs
    tidx = findall(x->x in red_times,times)
    ptidx = vec(transpose(collect(0:nparams-1)*time_ndofs .+ tidx'))
    NonaffinePTArray(sols[ptidx])
  else
    sols
  end
end

function old_mdeim_solve!(cache,ad::RBAffineDecomposition,a::Matrix;st_mdeim=false)
  csolve,crecast = cache
  time_ndofs = length(ad.integration_domain.times)
  nparams = Int(size(a,2)/time_ndofs)
  coeff = if st_mdeim
    _coeff = mdeim_solve!(csolve,ad.mdeim_interpolation,reshape(a,:,nparams))
    recast_coefficient!(crecast,first(ad.basis_time),_coeff)
  else
    _coeff = mdeim_solve!(csolve,ad.mdeim_interpolation,a)
    recast_coefficient!(crecast,_coeff)
  end
  return coeff
end

function collect_residuals_for_idx!(
  b::PTArray,
  op::PTOperator,
  sols::PTArray{T},
  nonzero_idx::Vector{Int},
  args...) where T

  ress = residual_for_idx!(b,op,sols,args...)
  N = length(ress)
  resmat = zeros(eltype(T),length(nonzero_idx),N)
  @inbounds for n = 1:N
    resmat[:,n] = ress[n][nonzero_idx]
  end
  return resmat
end

function residual_for_idx!(
  b::PTArray,
  op::PTThetaAffineMethodOperator,
  ::PTArray,
  args...)

  vθ = op.vθ
  z = zero(eltype(b))
  fill!(b,z)
  residual!(b,op.odeop,op.μ,op.tθ,(vθ,vθ),op.ode_cache,args...)
end

function collect_jacobians_for_idx!(
  A::PTArray,
  op::PTOperator,
  sols::PTArray{T},
  nonzero_idx::Vector{Int},
  args...;
  i=1) where T

  jacs_i = jacobian_for_idx!(A,op,sols,i,args...)
  N = length(jacs_i)
  jacimat = zeros(eltype(T),length(nonzero_idx),N)
  @inbounds for n = 1:N
    jacimat[:,n] = jacs_i[n][nonzero_idx].nzval
  end
  return jacimat
end

function jacobian_for_idx!(
  A::PTArray,
  op::PTThetaAffineMethodOperator,
  ::PTArray,
  i::Int,
  args...)

  vθ = op.vθ
  z = zero(eltype(A))
  fillstored!(A,z)
  jacobian!(A,op.odeop,op.μ,op.tθ,(vθ,vθ),i,(1.0,1/op.dtθ)[i],op.ode_cache,args...)
end

function residual!(
  b::PTArray,
  op::PTFEOperator,
  μ::AbstractVector,
  t::T,
  xh::S,
  cache,
  meas::Measure) where {T,S}

  V = get_test(op)
  v = get_fe_basis(V)
  res = get_residual(op)
  dc = integrate(res(μ,t,xh,v),meas)
  vecdata = collect_cell_vector(V,dc)
  assemble_vector_add!(b,op.assem,vecdata)
  b
end

function jacobian!(
  A::PTArray,
  op::PTFEOperator,
  μ::AbstractVector,
  t::T,
  uh::S,
  i::Integer,
  γᵢ::Real,
  cache,
  meas::Measure) where {T,S}

  Uh = get_trial(op)(μ,t)
  V = get_test(op)
  u = get_trial_fe_basis(Uh)
  v = get_fe_basis(V)
  jac = get_jacobian(op)
  dc = γᵢ*integrate(jac[i](μ,t,uh,u,v),meas)
  matdata = collect_cell_matrix(Uh,V,dc)
  assemble_matrix_add!(A,op.assem,matdata)
  A
end

function old_allocate_cache(
  op::PTOperator,
  snaps::PTArray{Vector{T}}) where T

  b = allocate_residual(op,snaps)
  A = allocate_jacobian(op,snaps)

  coeff = zeros(T,1,1)
  ptcoeff = NonaffinePTArray([zeros(T,1,1) for _ = eachindex(op.μ)])

  res_contrib_cache = return_cache(RBVecContributionMap(),snaps)
  jac_contrib_cache = return_cache(RBMatContributionMap(),snaps)

  res_cache = (b,CachedArray(coeff),CachedArray(ptcoeff)),res_contrib_cache
  jac_cache = (A,CachedArray(coeff),CachedArray(ptcoeff)),jac_contrib_cache
  res_cache,jac_cache
end

begin
  root = pwd()
  mesh = "model_circle_2D_coarse.json"
  bnd_info = Dict("dirichlet" => ["inlet","outlet"],"neumann" => ["noslip"])
  # mesh = "cube2x2.json"
  # bnd_info = Dict("dirichlet" => [1,2,3,4,5,7,8],"neumann" => [6])
  test_path = "$root/tests/poisson/unsteady/$mesh"
  order = 1
  degree = 2
  model = get_discrete_model(test_path,mesh,bnd_info)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=["neumann"])
  dΓn = Measure(Γn,degree)

  ranges = fill([1.,10.],3)
  sampling = UniformSampling()
  pspace = PSpace(ranges,sampling)

  a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
  a(μ,t) = x->a(x,μ,t)
  aμt(μ,t) = PTFunction(a,μ,t)

  f(x,μ,t) = 1.
  f(μ,t) = x->f(x,μ,t)
  fμt(μ,t) = PTFunction(f,μ,t)

  h(x,μ,t) = abs(cos(t/μ[3]))
  h(μ,t) = x->h(x,μ,t)
  hμt(μ,t) = PTFunction(h,μ,t)

  g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
  g(μ,t) = x->g(x,μ,t)

  u0(x,μ) = 0
  u0(μ) = x->u0(x,μ)
  u0μ(μ) = PFunction(u0,μ)

  res(μ,t,u,v) = ∫ₚ(v*∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⋅∇(u),dΩ) - ∫ₚ(fμt(μ,t)*v,dΩ) - ∫ₚ(hμt(μ,t)*v,dΓn)
  jac(μ,t,u,du,v) = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
  jac_t(μ,t,u,dut,v) = ∫ₚ(v*dut,dΩ)

  T = Float
  reffe = ReferenceFE(lagrangian,T,order)
  test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial = PTTrialFESpace(test,g)
  feop = AffinePTFEOperator(res,jac,jac_t,pspace,trial,test)
  t0,tf,dt,θ = 0.,0.3,0.005,1
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
  fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)

  ϵ = 1e-4
  load_solutions = false
  save_solutions = true
  load_structures = false
  save_structures = false
  norm_style = :l2
  nsnaps_state = 50
  nsnaps_mdeim = 20
  nsnaps_test = 10
  st_mdeim = false
  rbinfo = RBInfo(test_path;ϵ,norm_style,nsnaps_state,nsnaps_mdeim,nsnaps_test,st_mdeim)

  # sols,params = load(rbinfo,(Snapshots{Vector{T}},Table))
  # rbspace = load(rbinfo,RBSpace{T})
  params = realization(feop,nsnaps_state+nsnaps_test)
  sols,stats = collect_single_field_solutions(fesolver,feop,params)
  rbspace = reduced_basis(rbinfo,feop,sols)
  if save_solutions
    save(rbinfo,(sols,params,stats))
  end
end

rbrhs,rblhs = collect_compress_rhs_lhs(rbinfo,feop,fesolver,rbspace,params)
old_rbrhs,old_rblhs = old_collect_compress_rhs_lhs(rbinfo,feop,fesolver,rbspace,params)

println("Comparing linear RB problems")
nsnaps_test = rbinfo.nsnaps_test
snaps_test,params_test = sols[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
op = get_ptoperator(fesolver,feop,snaps_test,params_test)
old_cache = old_allocate_cache(op,snaps_test)
old_stats = @timed begin
  old_rhs,(old_lhs,old_lhs_t) = collect_rhs_lhs_contributions!(old_cache,rbinfo,op,old_rbrhs,old_rblhs,rbspace)
  old_rb_snaps_test = rb_solve(fesolver.nls,old_rhs,old_lhs+old_lhs_t)
end
old_approx_snaps_test = recast(old_rb_snaps_test,rbspace)

cache = allocate_cache(op,snaps_test)
stats = @timed begin
  rhs,(lhs,lhs_t) = collect_rhs_lhs_contributions!(cache,rbinfo,op,rbrhs,rblhs,rbspace)
  rb_snaps_test = rb_solve(fesolver.nls,rhs,lhs+lhs_t)
end
approx_snaps_test = recast(rb_snaps_test,rbspace)

old_approx_snaps_mat = space_time_matrices(old_approx_snaps_test;nparams=nsnaps_test)
approx_snaps_mat = space_time_matrices(approx_snaps_test;nparams=nsnaps_test)
snaps_mat = space_time_matrices(snaps_test;nparams=nsnaps_test)
post_process(rbinfo,feop,fesolver,snaps_test,params_test,old_approx_snaps_test,stats)

# background check
old_approx_snaps_mat[1] - snaps_mat[1]

# compare rb contribs
stack((old_rhs - rhs).array)
stack((old_lhs - lhs).array)
stack((old_lhs_t - lhs_t).array)

# compare affine decomp
trians_res = get_domains(old_rbrhs)
old_rbrhs1 = old_rbrhs[[trians_res...][1]]
rbrhs1 = rbrhs.affine_decompositions[1]

old_rbrhs1.basis_space - rbrhs1.basis_space
old_rbrhs1.basis_time - rbrhs1.basis_time
old_rbrhs1.integration_domain.idx - rbrhs1.integration_domain.idx
old_rbrhs1.mdeim_interpolation - rbrhs1.mdeim_interpolation

old_rblhs2 = old_rblhs[2][[get_domains(old_rblhs[2])...][1]]
rblhs2 = rblhs[2].affine_decompositions[1]
old_rblhs2.basis_space - rblhs2.basis_space
old_rblhs2.basis_time - rblhs2.basis_time
old_rblhs2.integration_domain.idx - rblhs2.integration_domain.idx
old_rblhs2.mdeim_interpolation - rblhs2.mdeim_interpolation
