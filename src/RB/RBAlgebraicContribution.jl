struct RBAlgebraicContribution{T,N}
  affine_decompositions::Vector{RBAffineDecomposition{T,N}}
  function RBAlgebraicContribution(
    affine_decompositions::Vector{<:RBAffineDecomposition{T,N}}
    ) where {T,N}
    new{T,N}(affine_decompositions)
  end
end

const RBVecAlgebraicContribution{T} = RBAlgebraicContribution{T,1}
const RBMatAlgebraicContribution{T} = RBAlgebraicContribution{T,2}

Base.length(a::RBAlgebraicContribution) = length(a.affine_decompositions)
Base.getindex(a::RBAlgebraicContribution,i::Int) = a.affine_decompositions[i]
Base.eachindex(a::RBAlgebraicContribution) = eachindex(a.affine_decompositions)
Base.iterate(a::RBAlgebraicContribution,i...) = iterate(a.affine_decompositions,i...)
Base.eltype(::RBAlgebraicContribution{T,N} where N) where T = T
Base.ndims(::RBAlgebraicContribution{T,N} where T) where N = N
Base.isempty(a::RBAlgebraicContribution) = isempty(a.affine_decompositions)
CellData.get_domains(a::RBAlgebraicContribution) = map(get_integration_domain,a.affine_decompositions)

function get_rb_ndofs(a::RBAlgebraicContribution)
  trian = first([get_domains(a)...])
  get_rb_ndofs(a[trian])
end

function save_algebraic_contrib(path::String,a::RBAlgebraicContribution)
  create_dir(path)
  adpath = joinpath(path,"ad")
  for (i,ai) in enumerate(a)
    save(adpath*"_$i",ai)
  end
end

function load_algebraic_contrib(path::String,::Type{RBVecAlgebraicContribution{T}},args...) where T
  S = RBVecAffineDecomposition{T}
  adpath = joinpath(path,"ad")
  a = S[]
  i = 1
  while isfile(correct_path(adpath*"_$i"))
    _ai = load(adpath*"_$i",S)
    ai = correct_measure(_ai,args...)
    push!(a,ai)
    i += 1
  end
  RBAlgebraicContribution(a)
end

function load_algebraic_contrib(path::String,::Type{RBMatAlgebraicContribution{T}},args...) where T
  S = RBMatAffineDecomposition{T}
  adpath = joinpath(path,"ad")
  a = S[]
  i = 1
  while isfile(correct_path(adpath*"_$i"))
    _ai = load(adpath*"_$i",S)
    ai = correct_measure(_ai,args...)
    push!(a,ai)
    i += 1
  end
  RBAlgebraicContribution(a)
end

function save(rbinfo::RBInfo,a::RBVecAlgebraicContribution)
  path = joinpath(rbinfo.rb_path,"rb_rhs")
  save_algebraic_contrib(path,a)
end

function load(rbinfo::RBInfo,::Type{RBVecAlgebraicContribution{T}},args...) where T
  S = RBVecAlgebraicContribution{T}
  path = joinpath(rbinfo.rb_path,"rb_rhs")
  load_algebraic_contrib(path,S,args...)
end

function save(rbinfo::RBInfo,a::Vector{<:RBMatAlgebraicContribution})
  for i = eachindex(a)
    path = joinpath(rbinfo.rb_path,"rb_lhs_$i")
    save_algebraic_contrib(path,a[i])
  end
end

function load(rbinfo::RBInfo,::Type{Vector{RBMatAlgebraicContribution{T}}},args...) where T
  S = RBMatAlgebraicContribution{T}
  njacs = num_active_dirs(rbinfo.rb_path)
  ad_jacs = Vector{S}(undef,njacs)
  for i = 1:njacs
    path = joinpath(rbinfo.rb_path,"rb_lhs_$i")
    ad_jacs[i] = load_algebraic_contrib(path,S,args...)
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

function load(rbinfo::RBInfo,::Type{NTuple{2,RBVecAlgebraicContribution{T}}},args...) where T
  S = RBVecAlgebraicContribution{T}
  path_lin = joinpath(rbinfo.rb_path,"rb_rhs_lin")
  path_nlin = joinpath(rbinfo.rb_path,"rb_rhs_nlin")
  a_lin = load_algebraic_contrib(path_lin,S,args...)
  a_nlin = load_algebraic_contrib(path_nlin,S,args...)
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

function load(rbinfo::RBInfo,::Type{NTuple{3,Vector{RBMatAlgebraicContribution{T}}}},args...) where T
  S = RBMatAlgebraicContribution{T}
  njacs = num_active_dirs(rbinfo.rb_path)
  ad_jacs_lin = Vector{S}(undef,njacs)
  ad_jacs_nlin = Vector{S}(undef,njacs)
  ad_jacs_aux = Vector{S}(undef,njacs)
  for i = 1:njacs
    path_lin = joinpath(rbinfo.rb_path,"rb_lhs_lin_$i")
    path_nlin = joinpath(rbinfo.rb_path,"rb_lhs_nlin_$i")
    path_aux = joinpath(rbinfo.rb_path,"rb_lhs_aux_$i")
    ad_jacs_lin[i] = load_algebraic_contrib(path_lin,S,args...)
    ad_jacs_nlin[i] = load_algebraic_contrib(path_nlin,S,args...)
    ad_jacs_aux[i] = load_algebraic_contrib(path_aux,S,args...)
  end
  ad_jacs_lin,ad_jacs_nlin,ad_jacs_aux
end

function collect_compress_rhs_lhs(rbinfo,feop::PTFEOperator{Affine},fesolver,rbspace,params)
  μ = params[1:rbinfo.nsnaps_mdeim]
  op = get_ptoperator(fesolver,feop,rbspace,μ)

  println("Computing RB affine decomposition (linear)")
  rhs = collect_compress_rhs(rbinfo,op,rbspace)
  lhs = collect_compress_lhs(rbinfo,op,rbspace;θ=fesolver.θ)

  return rhs,lhs
end

function collect_compress_rhs_lhs(rbinfo,feop,fesolver::PThetaMethod,rbspace,params)
  μ = params[1:rbinfo.nsnaps_mdeim]
  op = get_ptoperator(fesolver,feop,rbspace,μ)

  println("Computing RB affine decomposition (linear)")
  op_lin = linear_operator(op)
  rhs_lin = collect_compress_rhs(rbinfo,op_lin,rbspace)
  lhs_lin = collect_compress_lhs(rbinfo,op_lin,rbspace;θ=fesolver.θ)
  println("Computing RB affine decomposition (nonlinear)")
  op_nlin = nonlinear_operator(op)
  rhs_nlin = collect_compress_rhs(rbinfo,op_nlin,rbspace)
  lhs_nlin = collect_compress_lhs(rbinfo,op_nlin,rbspace;θ=fesolver.θ)
  println("Computing RB affine decomposition (auxiliary)")
  op_aux = auxiliary_operator(op)
  rblhs_aux = collect_compress_lhs(rbinfo,op_aux,rbspace;θ=fesolver.θ)

  rhs = rhs_lin,rhs_nlin
  lhs = lhs_lin,lhs_nlin,rblhs_aux

  return rhs,lhs
end

function collect_compress_rhs(
  rbinfo::RBInfo,
  op::PTOperator,
  rbspace::RBSpace{T}) where T

  ress,trian = collect_residuals_for_trian(op)
  compress_component(rbinfo,op,ress,trian,rbspace)
end

function collect_compress_lhs(
  rbinfo::RBInfo,
  op::PTOperator,
  rbspace::RBSpace{T};
  kwargs...) where T

  njacs = length(op.odeop.feop.jacs)
  ad_jacs = Vector{RBMatAlgebraicContribution{T}}(undef,njacs)
  for i = 1:njacs
    ad_jacs[i] = _collect_compress_lhs(rbinfo,op,rbspace,rbspace;i,kwargs...)
  end
  return ad_jacs
end

function _collect_compress_lhs(
  rbinfo::RBInfo,
  op::PTOperator,
  rbspace_row::RBSpace,
  rbspace_col::RBSpace;
  i=1,θ=1)

  combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*x-θ*y
  jacs,trian = collect_jacobians_for_trian(op;i)
  compress_component(rbinfo,op,jacs,trian,rbspace_row,rbspace_col;combine_projections)
end

function compress_component(
  rbinfo::RBInfo,
  op::PTOperator,
  snaps::Vector{<:NnzMatrix},
  trian::Base.KeySet{Triangulation},
  args...;
  kwargs...)

  affine_decompositions = map(zip(snaps,trian)) do (si,ti)
    RBAffineDecomposition(rbinfo,op,si,ti,args...;kwargs...)
  end
  return RBAlgebraicContribution(affine_decompositions)
end

function collect_rhs_lhs_contributions!(cache,rbinfo,op,rbres,rbjacs,rbspace)
  rhs_cache,lhs_cache = cache
  rhs = collect_rhs_contributions!(rhs_cache,rbinfo,op,rbres,rbspace)
  lhs = collect_lhs_contributions!(lhs_cache,rbinfo,op,rbjacs,rbspace)
  return rhs,lhs
end

function collect_reduced_residuals!(cache,op::PTOperator,rbres::RBVecAlgebraicContribution)
  collect_reduced_residuals!(cache,op,rbres.affine_decompositions)
end

function collect_reduced_jacobians!(cache,op::PTOperator,rbres::RBMatAlgebraicContribution;kwargs...)
  collect_reduced_jacobians!(cache,op,rbres.affine_decompositions;kwargs...)
end

function collect_rhs_contributions!(
  cache,
  rbinfo::RBInfo,
  op::PTOperator,
  rbres::RBVecAlgebraicContribution{T},
  rbspace::RBSpace{T}) where T

  mdeim_cache,rb_cache = cache
  st_mdeim = rbinfo.st_mdeim
  k = RBVecContributionMap()
  if isempty(rbres)
    return empty_rb_contribution(k,rbinfo,rbspace)
  else
    collect_cache,coeff_cache = mdeim_cache
    res = collect_reduced_residuals!(collect_cache,op,rbres)
    rb_res_contribs = Vector{PTArray{Vector{T}}}(undef,length(rbres))
    for i = eachindex(rbres)
      coeff = rb_coefficient!(coeff_cache,rbres[i],res[i];st_mdeim)
      rb_res_contribs[i] = rb_contribution!(rb_cache,k,rbres[i],coeff)
    end
  end
  return sum(rb_res_contribs)
end

function collect_lhs_contributions!(
  cache,
  rbinfo::RBInfo,
  op::PTOperator,
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
  op::PTOperator,
  rbjac::RBMatAlgebraicContribution{T},
  rbspace_row::RBSpace{T},
  rbspace_col::RBSpace{T};
  kwargs...) where T

  mdeim_cache,rb_cache = cache
  st_mdeim = rbinfo.st_mdeim
  k = RBMatContributionMap()
  if isempty(rbjac)
    return empty_rb_contribution(k,rbinfo,rbspace_row,rbspace_col)
  else
    collect_cache,coeff_cache = mdeim_cache
    jac = collect_reduced_jacobians!(collect_cache,op,rbjac;kwargs...)
    rb_jac_contribs = Vector{PTArray{Matrix{T}}}(undef,length(rbjac))
    for i = eachindex(rbjac)
      coeff = rb_coefficient!(coeff_cache,rbjac[i],jac[i];st_mdeim)
      rb_jac_contribs[i] = rb_contribution!(rb_cache,k,rbjac[i],coeff)
    end
  end
  return sum(rb_jac_contribs)
end
