abstract type RBAlgebraicContribution{T,N} end

struct RBVecAlgebraicContribution{T} <: RBAlgebraicContribution{T,1}
  dict::IdDict{Triangulation,RBVecAffineDecomposition{T}}
  function RBVecAlgebraicContribution(::Type{T}) where T
    new{T}(IdDict{Triangulation,RBVecAffineDecomposition{T}}())
  end
end

struct RBMatAlgebraicContribution{T} <: RBAlgebraicContribution{T,2}
  dict::IdDict{Triangulation,RBMatAffineDecomposition{T}}
  function RBMatAlgebraicContribution(::Type{T}) where T
    new{T}(IdDict{Triangulation,RBMatAffineDecomposition{T}}())
  end
end

CellData.num_domains(a::RBAlgebraicContribution) = length(a.dict)
CellData.get_domains(a::RBAlgebraicContribution) = keys(a.dict)
Base.iszero(a::RBAlgebraicContribution) = num_domains(a) == 0

function CellData.get_contribution(
  a::RBAlgebraicContribution,
  trian::Triangulation)

  if haskey(a.dict,trian)
    return a.dict[trian]
  else
    @unreachable """\n
    There is not contribution associated with the given mesh in this RBAlgebraicContribution object.
    """
  end
end

Base.getindex(a::RBAlgebraicContribution,trian::Triangulation) = get_contribution(a,trian)
Base.eltype(::RBAlgebraicContribution{T,N} where N) where T = T

function CellData.add_contribution!(
  a::RBAlgebraicContribution,
  trian::Triangulation,
  b)

  @check !haskey(a.dict,trian)
  a.dict[trian] = b
  a
end

function get_rb_ndofs(a::RBAlgebraicContribution)
  trian = first([get_domains(a)...])
  get_rb_ndofs(a[trian])
end

function save_algebraic_contrib(path::String,a::RBAlgebraicContribution{T,N} where N) where T
  create_dir!(path)
  cpath = joinpath(path,"contrib")
  tpath = joinpath(path,"trian")
  for (i,trian) in enumerate(get_domains(a))
    ai = a[trian]
    save(cpath*"_$i",ai)
    save(tpath*"_$i",trian)
  end
end

function load_algebraic_contrib(path::String,::Type{RBVecAlgebraicContribution{T}}) where T
  cpath = joinpath(path,"contrib")
  tpath = joinpath(path,"trian")
  a = RBVecAlgebraicContribution(T)
  i = 1
  while isfile(correct_path(cpath*"_$i"))
    ai = load(cpath*"_$i",RBVecAffineDecomposition{T})
    ti = load(tpath*"_$i",Triangulation)
    add_contribution!(a,ti,ai)
    i += 1
  end
  a
end

function load_algebraic_contrib(path::String,::Type{RBMatAlgebraicContribution{T}}) where T
  cpath = joinpath(path,"contrib")
  tpath = joinpath(path,"trian")
  a = RBMatAlgebraicContribution(T)
  i = 1
  while isfile(correct_path(cpath*"_$i"))
    ai = load(cpath*"_$i",RBMatAffineDecomposition{T})
    ti = load(tpath*"_$i",Triangulation)
    add_contribution!(a,ti,ai)
    i += 1
  end
  a
end

function save(info::RBInfo,a::RBVecAlgebraicContribution)
  path = joinpath(info.rb_path,"rb_rhs")
  save_algebraic_contrib(path,a)
end

function load(info::RBInfo,::Type{RBVecAlgebraicContribution{T}}) where T
  path = joinpath(info.rb_path,"rb_rhs")
  load_algebraic_contrib(path,RBVecAlgebraicContribution{T})
end

function save(info::RBInfo,a::NTuple{2,RBVecAlgebraicContribution})
  a_lin,a_nlin = a
  path_lin = joinpath(info.rb_path,"rb_rhs_lin")
  path_nlin = joinpath(info.rb_path,"rb_rhs_nlin")
  save_algebraic_contrib(path_lin,a_lin)
  save_algebraic_contrib(path_nlin,a_nlin)
end

function load(info::RBInfo,::Type{NTuple{2,RBVecAlgebraicContribution{T}}}) where T
  path_lin = joinpath(info.rb_path,"rb_rhs_lin")
  path_nlin = joinpath(info.rb_path,"rb_rhs_nlin")
  a_lin = load_algebraic_contrib(path_lin,RBVecAlgebraicContribution{T})
  a_nlin = load_algebraic_contrib(path_nlin,RBVecAlgebraicContribution{T})
  a_lin,a_nlin
end

function save(info::RBInfo,a::NTuple{3,Vector{<:RBMatAlgebraicContribution}})
  a_lin,a_nlin,a_aux = a
  for i = eachindex(a_lin)
    path_lin = joinpath(info.rb_path,"rb_lhs_lin_$i")
    path_nlin = joinpath(info.rb_path,"rb_lhs_nlin_$i")
    path_aux = joinpath(info.rb_path,"rb_lhs_aux_$i")
    save_algebraic_contrib(path_lin,a_lin[i])
    save_algebraic_contrib(path_nlin,a_nlin[i])
    save_algebraic_contrib(path_aux,a_aux[i])
  end
end

function load(info::RBInfo,::Type{NTuple{3,Vector{RBMatAlgebraicContribution{T}}}}) where T
  njacs = num_active_dirs(info.rb_path)
  ad_jacs_lin = Vector{RBMatAlgebraicContribution{T}}(undef,njacs)
  ad_jacs_nlin = Vector{RBMatAlgebraicContribution{T}}(undef,njacs)
  ad_jacs_aux = Vector{RBMatAlgebraicContribution{T}}(undef,njacs)
  for i = 1:njacs
    path_lin = joinpath(info.rb_path,"rb_lhs_lin_$i")
    path_nlin = joinpath(info.rb_path,"rb_lhs_nlin_$i")
    path_aux = joinpath(info.rb_path,"rb_lhs_aux_$i")
    ad_jacs_lin[i] = load_algebraic_contrib(path_lin,RBMatAlgebraicContribution{T})
    ad_jacs_nlin[i] = load_algebraic_contrib(path_nlin,RBMatAlgebraicContribution{T})
    ad_jacs_aux[i] = load_algebraic_contrib(path_aux,RBMatAlgebraicContribution{T})
  end
  ad_jacs_lin,ad_jacs_nlin,ad_jacs_aux
end

function save(info::RBInfo,a::Vector{<:RBMatAlgebraicContribution})
  for i = eachindex(a)
    path = joinpath(info.rb_path,"rb_lhs_$i")
    save_algebraic_contrib(path,a[i])
  end
end

function load(info::RBInfo,::Type{Vector{RBMatAlgebraicContribution{T}}}) where T
  njacs = num_active_dirs(info.rb_path)
  ad_jacs = Vector{RBMatAlgebraicContribution{T}}(undef,njacs)
  for i = 1:njacs
    path = joinpath(info.rb_path,"rb_lhs_$i")
    ad_jacs[i] = load_algebraic_contrib(path,RBMatAlgebraicContribution{T})
  end
  ad_jacs
end

function collect_compress_rhs_lhs(
  info::RBInfo,
  feop::PTFEOperator{Affine},
  fesolver::PThetaMethod,
  rbspace,
  params::Table)

  θ = fesolver.θ
  nsnaps_mdeim = info.nsnaps_mdeim
  μ = params[1:nsnaps_mdeim]
  op = get_ptoperator(fesolver,feop,rbspace,μ)

  rhs = collect_compress_rhs(info,op,rbspace)
  lhs = collect_compress_lhs(info,op,rbspace;θ)

  return rhs,lhs
end

function collect_compress_rhs_lhs(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbspace,
  params::Table)

  θ = fesolver.θ
  nsnaps_mdeim = info.nsnaps_mdeim
  μ = params[1:nsnaps_mdeim]
  op = get_ptoperator(fesolver,feop,rbspace,μ)

  op_lin = linear_operator(op)
  rhs_lin = collect_compress_rhs(info,op_lin,rbspace)
  lhs_lin = collect_compress_lhs(info,op_lin,rbspace;θ)
  op_nlin = nonlinear_operator(op)
  rhs_nlin = collect_compress_rhs(info,op_nlin,rbspace)
  lhs_nlin = collect_compress_lhs(info,op_nlin,rbspace;θ)
  op_aux = auxiliary_operator(op)
  rblhs_aux = collect_compress_lhs(info,op_aux,rbspace;θ)

  rhs = rhs_lin,rhs_nlin
  lhs = lhs_lin,lhs_nlin,rblhs_aux

  return rhs,lhs
end

function collect_compress_rhs(
  info::RBInfo,
  op::PTAlgebraicOperator,
  rbspace::RBSpace{T}) where T

  ress,trian = collect_residuals_for_trian(op)
  ad_res = RBVecAlgebraicContribution(T)
  compress_component!(ad_res,info,op,ress,trian,rbspace)
  return ad_res
end

function collect_compress_lhs(
  info::RBInfo,
  op::PTAlgebraicOperator,
  rbspace::RBSpace{T};
  θ::Real=1) where T

  njacs = length(op.odeop.feop.jacs)
  ad_jacs = Vector{RBMatAlgebraicContribution{T}}(undef,njacs)
  for i = 1:njacs
    combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*x-θ*y
    jacs,trian = collect_jacobians_for_trian(op;i)
    ad_jac_i = RBMatAlgebraicContribution(T)
    compress_component!(ad_jac_i,info,op,jacs,trian,rbspace,rbspace;combine_projections)
    ad_jacs[i] = ad_jac_i
  end
  return ad_jacs
end

function compress_component!(
  contrib::RBAlgebraicContribution,
  info::RBInfo,
  op::PTAlgebraicOperator,
  snaps::Vector{<:NnzMatrix},
  trian::Base.KeySet{Triangulation},
  args...;
  kwargs...)

  for (i,ti) in enumerate(trian)
    si = snaps[i]
    if !iszero(si)
      ci = RBAffineDecomposition(info,op,si,ti,args...;kwargs...)
      add_contribution!(contrib,ti,ci)
    end
  end
end

function collect_rhs_lhs_contributions!(
  cache,
  info::RBInfo,
  op::PTAlgebraicOperator,
  rbres,
  rbjacs,
  rbspace)

  rhs_cache,lhs_cache = cache
  rhs = collect_rhs_contributions!(rhs_cache,info,op,rbres,rbspace)
  lhs = collect_lhs_contributions!(lhs_cache,info,op,rbjacs,rbspace)
  return rhs,lhs
end

function collect_rhs_contributions!(
  cache,
  info::RBInfo,
  op::PTAlgebraicOperator,
  rbres::RBVecAlgebraicContribution{T},
  rbspace::RBSpace{T}) where T

  coeff_cache,rb_cache = cache
  st_mdeim = info.st_mdeim
  k = RBVecContributionMap(T)
  rb_res_contribs = Vector{PTArray{Vector{T}}}(undef,num_domains(rbres))
  if iszero(rbres)
    nrow = get_rb_ndofs(rbspace)
    contrib = PTArray([zeros(T,nrow) for _ = eachindex(op.μ)])
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
  info::RBInfo,
  op::PTAlgebraicOperator,
  rbjacs::Vector{RBMatAlgebraicContribution{T}},
  rbspace::RBSpace{T}) where T

  njacs = length(rbjacs)
  rb_jacs_contribs = Vector{PTArray{Matrix{T}}}(undef,njacs)
  for i = 1:njacs
    rb_jac_i = rbjacs[i]
    rb_jacs_contribs[i] = collect_lhs_contributions!(cache,info,op,rb_jac_i,rbspace,rbspace;i)
  end
  return rb_jacs_contribs
end

function collect_lhs_contributions!(
  cache,
  info::RBInfo,
  op::PTAlgebraicOperator,
  rbjac::RBMatAlgebraicContribution{T},
  rbspace_row::RBSpace{T},
  rbspace_col::RBSpace{T};
  kwargs...) where T

  coeff_cache,rb_cache = cache
  trian = get_domains(rbjac)
  st_mdeim = info.st_mdeim
  k = RBMatContributionMap(T)
  rb_jac_contribs = Vector{PTArray{Matrix{T}}}(undef,num_domains(rbjac))
  if iszero(rbjac)
    nrow = get_rb_ndofs(rbspace_row)
    ncol = get_rb_ndofs(rbspace_col)
    contrib = PTArray([zeros(T,nrow,ncol) for _ = eachindex(op.μ)])
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
