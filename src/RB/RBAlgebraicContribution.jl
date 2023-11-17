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

function update_reduced_operator!(a::RBAlgebraicContribution,args...)
  for trian in get_domains(a)
    update_reduced_operator!(a[trian],args...)
  end
end

function update_reduced_operator!(a::Vector{<:RBAlgebraicContribution},args...)
  for ai in a
    update_reduced_operator!(ai,args...)
  end
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

function save(rbinfo::RBInfo,a::RBVecAlgebraicContribution)
  path = joinpath(rbinfo.rb_path,"rb_rhs")
  save_algebraic_contrib(path,a)
end

function load(rbinfo::RBInfo,T::Type{RBVecAlgebraicContribution{S}}) where S
  path = joinpath(rbinfo.rb_path,"rb_rhs")
  load_algebraic_contrib(path,T)
end

function save(rbinfo::RBInfo,a::Vector{<:RBMatAlgebraicContribution})
  for i = eachindex(a)
    path = joinpath(rbinfo.rb_path,"rb_lhs_$i")
    save_algebraic_contrib(path,a[i])
  end
end

function load(rbinfo::RBInfo,T::Type{Vector{RBMatAlgebraicContribution{S}}}) where S
  njacs = num_active_dirs(rbinfo.rb_path)
  ad_jacs = Vector{RBMatAlgebraicContribution{S}}(undef,njacs)
  for i = 1:njacs
    path = joinpath(rbinfo.rb_path,"rb_lhs_$i")
    ad_jacs[i] = load_algebraic_contrib(path,T)
  end
  ad_jacs
end

function save(rbinfo::RBInfo,a::NTuple{2,RBVecAlgebraicContribution})
  a_lin,a_nlin = a
  path_lin = joinpath(rbinfo.rb_path,"rb_rhs_lin")
  path_nlin = joinpath(rbinfo.rb_path,"rb_rhs_nlin")
  save_algebraic_contrib(path_lin,a_lin)
  save_algebraic_contrib(path_nlin,a_nlin)
end

function load(rbinfo::RBInfo,T::Type{NTuple{2,RBVecAlgebraicContribution{S}}}) where S
  path_lin = joinpath(rbinfo.rb_path,"rb_rhs_lin")
  path_nlin = joinpath(rbinfo.rb_path,"rb_rhs_nlin")
  a_lin = load_algebraic_contrib(path_lin,T)
  a_nlin = load_algebraic_contrib(path_nlin,T)
  a_lin,a_nlin
end

function save(rbinfo::RBInfo,a::NTuple{3,Vector{<:RBMatAlgebraicContribution}})
  a_lin,a_nlin,a_aux = a
  for i = eachindex(a_lin)
    path_lin = joinpath(rbinfo.rb_path,"rb_lhs_lin_$i")
    path_nlin = joinpath(rbinfo.rb_path,"rb_lhs_nlin_$i")
    path_aux = joinpath(rbinfo.rb_path,"rb_lhs_aux_$i")
    save_algebraic_contrib(path_lin,a_lin[i])
    save_algebraic_contrib(path_nlin,a_nlin[i])
    save_algebraic_contrib(path_aux,a_aux[i])
  end
end

function load(rbinfo::RBInfo,T::Type{NTuple{3,Vector{RBMatAlgebraicContribution{S}}}}) where S
  njacs = num_active_dirs(rbinfo.rb_path)
  ad_jacs_lin = Vector{RBMatAlgebraicContribution{S}}(undef,njacs)
  ad_jacs_nlin = Vector{RBMatAlgebraicContribution{S}}(undef,njacs)
  ad_jacs_aux = Vector{RBMatAlgebraicContribution{S}}(undef,njacs)
  for i = 1:njacs
    path_lin = joinpath(rbinfo.rb_path,"rb_lhs_lin_$i")
    path_nlin = joinpath(rbinfo.rb_path,"rb_lhs_nlin_$i")
    path_aux = joinpath(rbinfo.rb_path,"rb_lhs_aux_$i")
    ad_jacs_lin[i] = load_algebraic_contrib(path_lin,T)
    ad_jacs_nlin[i] = load_algebraic_contrib(path_nlin,T)
    ad_jacs_aux[i] = load_algebraic_contrib(path_aux,T)
  end
  ad_jacs_lin,ad_jacs_nlin,ad_jacs_aux
end

function collect_compress_rhs_lhs(
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
  rhs = collect_compress_rhs(rbinfo,op,rbspace)
  lhs = collect_compress_lhs(rbinfo,op,rbspace;θ)

  return rhs,lhs
end

function collect_compress_rhs_lhs(
  rbinfo::RBInfo,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbspace,
  params::Table)

  θ = fesolver.θ
  nsnaps_mdeim = rbinfo.nsnaps_mdeim
  μ = params[1:nsnaps_mdeim]
  op = get_ptoperator(fesolver,feop,rbspace,μ)

  println("Computing RB affine decomposition (linear)")
  op_lin = linear_operator(op)
  rhs_lin = collect_compress_rhs(rbinfo,op_lin,rbspace)
  lhs_lin = collect_compress_lhs(rbinfo,op_lin,rbspace;θ)
  println("Computing RB affine decomposition (nonlinear)")
  op_nlin = nonlinear_operator(op)
  rhs_nlin = collect_compress_rhs(rbinfo,op_nlin,rbspace)
  lhs_nlin = collect_compress_lhs(rbinfo,op_nlin,rbspace;θ)
  println("Computing RB affine decomposition (auxiliary)")
  op_aux = auxiliary_operator(op)
  rblhs_aux = collect_compress_lhs(rbinfo,op_aux,rbspace;θ)

  rhs = rhs_lin,rhs_nlin
  lhs = lhs_lin,lhs_nlin,rblhs_aux

  return rhs,lhs
end

function collect_compress_rhs(
  rbinfo::RBInfo,
  op::PTAlgebraicOperator,
  rbspace::RBSpace{T}) where T

  ress,trian = collect_residuals_for_trian(op)
  ad_res = RBVecAlgebraicContribution(T)
  compress_component!(ad_res,rbinfo,op,ress,trian,rbspace)
  return ad_res
end

function collect_compress_lhs(
  rbinfo::RBInfo,
  op::PTAlgebraicOperator,
  rbspace::RBSpace{T};
  θ::Real=1) where T

  njacs = length(op.odeop.feop.jacs)
  ad_jacs = Vector{RBMatAlgebraicContribution{T}}(undef,njacs)
  for i = 1:njacs
    combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*x-θ*y
    jacs,trian = collect_jacobians_for_trian(op;i)
    ad_jac_i = RBMatAlgebraicContribution(T)
    compress_component!(ad_jac_i,rbinfo,op,jacs,trian,rbspace,rbspace;combine_projections)
    ad_jacs[i] = ad_jac_i
  end
  return ad_jacs
end

function compress_component!(
  contrib::RBAlgebraicContribution,
  rbinfo::RBInfo,
  op::PTAlgebraicOperator,
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

function collect_rhs_lhs_contributions!(cache,rbinfo,rbres,rbjacs,rbspace)
  rhs_cache,lhs_cache = cache
  rhs = collect_rhs_contributions!(rhs_cache,rbinfo,rbres,rbspace)
  lhs = collect_lhs_contributions!(lhs_cache,rbinfo,rbjacs,rbspace)
  return rhs,lhs
end

function collect_rhs_contributions!(
  cache,
  rbinfo::RBInfo,
  rbres::RBVecAlgebraicContribution{T},
  rbspace::RBSpace{T}) where T

  mdeim_cache,rb_cache = cache
  st_mdeim = rbinfo.st_mdeim
  k = RBVecContributionMap(T)
  rb_res_contribs = Vector{PTArray{Vector{T}}}(undef,num_domains(rbres))
  if iszero(rbres)
    op = get_ptoperator(rbres)
    nrow = get_rb_ndofs(rbspace)
    contrib = AffinePTArray(zeros(T,nrow),length(op.μ))
    rb_res_contribs[i] = contrib
  else
    for (i,t) in enumerate(get_domains(rbres))
      rbrest = rbres[t]
      coeff = rhs_coefficient!(mdeim_cache,rbrest;st_mdeim)
      rb_res_contribs[i] = rb_contribution!(rb_cache,k,rbrest,coeff)
    end
  end
  return sum(rb_res_contribs)
end

function collect_lhs_contributions!(
  cache,
  rbinfo::RBInfo,
  rbjacs::Vector{RBMatAlgebraicContribution{T}},
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
  op::PTAlgebraicOperator,
  rbjac::RBMatAlgebraicContribution{T},
  rbspace_row::RBSpace{T},
  rbspace_col::RBSpace{T};
  kwargs...) where T

  mdeim_cache,rb_cache = cache
  trian = get_domains(rbjac)
  st_mdeim = rbinfo.st_mdeim
  k = RBMatContributionMap(T)
  rb_jac_contribs = Vector{PTArray{Matrix{T}}}(undef,num_domains(rbjac))
  if iszero(rbjac)
    op = get_ptoperator(rbres)
    nrow = get_rb_ndofs(rbspace_row)
    ncol = get_rb_ndofs(rbspace_col)
    contrib = AffinePTArray(zeros(T,nrow,ncol),length(op.μ))
    rb_jac_contribs[i] = contrib
  else
    for (i,t) in enumerate(trian)
      rbjact = rbjac[t]
      coeff = lhs_coefficient!(mdeim_cache,rbjact;st_mdeim,kwargs...)
      rb_jac_contribs[i] = rb_contribution!(rb_cache,k,rbjact,coeff)
    end
  end
  return sum(rb_jac_contribs)
end
