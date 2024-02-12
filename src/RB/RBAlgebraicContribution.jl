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

function num_rb_ndofs(a::RBAlgebraicContribution)
  trian = first([get_domains(a)...])
  num_rb_ndofs(a[trian])
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
    ai = ReducedMeasure(_ai,args...)
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
    ai = ReducedMeasure(_ai,args...)
    push!(a,ai)
    i += 1
  end
  RBAlgebraicContribution(a)
end

function Utils.save(info::RBInfo,a::RBVecAlgebraicContribution)
  path = joinpath(info.rb_path,"rb_rhs")
  save_algebraic_contrib(path,a)
end

function Utils.load(info::RBInfo,::Type{RBVecAlgebraicContribution{T}},args...) where T
  S = RBVecAlgebraicContribution{T}
  path = joinpath(info.rb_path,"rb_rhs")
  load_algebraic_contrib(path,S,args...)
end

function Utils.save(info::RBInfo,a::Vector{<:RBMatAlgebraicContribution})
  map(eachindex(a)) do i
    path = joinpath(info.rb_path,"rb_lhs_$i")
    save_algebraic_contrib(path,a[i])
  end
end

function Utils.load(info::RBInfo,::Type{Vector{RBMatAlgebraicContribution{T}}},args...) where T
  S = RBMatAlgebraicContribution{T}
  njacs = num_active_dirs(info.rb_path)
  map(1:njacs) do i
    path = joinpath(info.rb_path,"rb_lhs_$i")
    load_algebraic_contrib(path,S,args...)
  end
end

function Utils.save(info::RBInfo,a::NTuple{2,RBVecAlgebraicContribution})
  a_lin,a_nlin = a
  path_lin = joinpath(info.rb_path,"rb_rhs_lin")
  path_nlin = joinpath(info.rb_path,"rb_rhs_nlin")
  save_algebraic_contrib(path_lin,a_lin)
  save_algebraic_contrib(path_nlin,a_nlin)
end

function Utils.load(info::RBInfo,::Type{NTuple{2,RBVecAlgebraicContribution{T}}},args...) where T
  S = RBVecAlgebraicContribution{T}
  path_lin = joinpath(info.rb_path,"rb_rhs_lin")
  path_nlin = joinpath(info.rb_path,"rb_rhs_nlin")
  a_lin = load_algebraic_contrib(path_lin,S,args...)
  a_nlin = load_algebraic_contrib(path_nlin,S,args...)
  a_lin,a_nlin
end

function Utils.save(info::RBInfo,a::NTuple{2,Vector{<:RBMatAlgebraicContribution}})
  a_lin,a_nlin = a
  map(eachindex(a_lin)) do i
    path_lin = joinpath(info.rb_path,"rb_lhs_lin_$i")
    path_nlin = joinpath(info.rb_path,"rb_lhs_nlin_$i")
    save_algebraic_contrib(path_lin,a_lin[i])
    save_algebraic_contrib(path_nlin,a_nlin[i])
  end
end

function Utils.load(info::RBInfo,::Type{NTuple{2,Vector{RBMatAlgebraicContribution{T}}}},args...) where T
  S = RBMatAlgebraicContribution{T}
  njacs = num_active_dirs(info.rb_path)
  map(1:njacs) do i
    path_lin = joinpath(info.rb_path,"rb_lhs_lin_$i")
    path_nlin = joinpath(info.rb_path,"rb_lhs_nlin_$i")
    ad_jacs_lin = load_algebraic_contrib(path_lin,S,args...)
    ad_jacs_nlin = load_algebraic_contrib(path_nlin,S,args...)
    ad_jacs_lin,ad_jacs_nlin
  end |> tuple_of_arrays
end

function collect_compress_rhs_lhs(info,feop::TransientParamFEOperator{Affine},fesolver,rbspace,params)
  μ = params[1:info.nsnaps_mdeim]
  op = get_method_operator(fesolver,feop,rbspace,μ)

  println("Computing RB affine decomposition (linear)")
  rhs = collect_compress_rhs(info,op,rbspace)
  lhs = collect_compress_lhs(info,op,rbspace;θ=fesolver.θ)

  return rhs,lhs
end

function collect_compress_rhs_lhs(info,feop_lin,feop_nonlin,fesolver,rbspace,params)
  μ = params[1:info.nsnaps_mdeim]

  println("Computing RB affine decomposition (linear)")
  op_lin = get_method_operator(fesolver,feop_lin,rbspace,μ)
  rhs_lin = collect_compress_rhs(info,op_lin,rbspace)
  lhs_lin = collect_compress_lhs(info,op_lin,rbspace;θ=fesolver.θ)
  println("Computing RB affine decomposition (nonlinear)")
  op_nlin = get_method_operator(fesolver,feop_nonlin,rbspace,μ)
  rhs_nlin = collect_compress_rhs(info,op_nlin,rbspace)
  lhs_nlin = collect_compress_lhs(info,op_nlin,rbspace;θ=fesolver.θ)

  rhs = rhs_lin,rhs_nlin
  lhs = lhs_lin,lhs_nlin

  return rhs,lhs
end

function collect_compress_rhs(
  info::RBInfo,
  op::NonlinearOperator,
  rbspace::RBSpace{T}) where T

  ress,trian = collect_residuals_for_trian(op)
  compress_component(info,op,ress,trian,rbspace)
end

function collect_compress_lhs(
  info::RBInfo,
  op::NonlinearOperator,
  rbspace::RBSpace;
  kwargs...)

  map(eachindex(op.feop.jacs)) do i
    _collect_compress_lhs(info,op,rbspace,rbspace;i,kwargs...)
  end
end

function _collect_compress_lhs(
  info::RBInfo,
  op::NonlinearOperator,
  rbspace_row::RBSpace,
  rbspace_col::RBSpace;
  i=1,θ=1)

  combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*x-θ*y
  jacs,trian = collect_jacobians_for_trian(op;i)
  compress_component(info,op,jacs,trian,rbspace_row,rbspace_col;combine_projections)
end

function compress_component(
  info::RBInfo,
  op::NonlinearOperator,
  snaps::Vector{<:NnzMatrix},
  trian::Base.KeySet{Triangulation},
  args...;
  kwargs...)

  affine_decompositions = map(zip(snaps,trian)) do (si,ti)
    RBAffineDecomposition(info,op,si,ti,args...;kwargs...)
  end
  return RBAlgebraicContribution(affine_decompositions)
end

function collect_rhs_lhs_contributions!(cache,info,op,rbres,rbjacs,rbspace)
  rhs_cache,lhs_cache = cache
  rhs = collect_rhs_contributions!(rhs_cache,info,op,rbres,rbspace)
  lhs = collect_lhs_contributions!(lhs_cache,info,op,rbjacs,rbspace)
  return rhs,lhs
end

function collect_rhs_contributions!(
  cache,
  info::RBInfo,
  op::NonlinearOperator,
  rbres::RBVecAlgebraicContribution,
  rbspace::RBSpace)

  mdeim_cache,rb_cache = cache
  st_mdeim = info.st_mdeim
  k = RBVecContributionMap()
  if isempty(rbres)
    zero_rb_contribution(k,info,rbspace)
  else
    collect_cache,coeff_cache = mdeim_cache
    res = collect_reduced_residuals!(collect_cache,op,rbres)
    rb_res_contribs = map(eachindex(rbres)) do i
      coeff = rb_coefficient!(coeff_cache,rbres[i],res[i];st_mdeim)
      rb_contribution!(rb_cache,k,rbres[i],coeff)
    end
    sum(rb_res_contribs)
  end
end

function collect_lhs_contributions!(
  cache,
  info::RBInfo,
  op::NonlinearOperator,
  rbjacs::Vector{<:RBMatAlgebraicContribution},
  rbspace::RBSpace)

  map(eachindex(rbjacs)) do i
    collect_lhs_contributions!(cache,info,op,rbjacs[i],rbspace,rbspace;i)
  end
end

function collect_lhs_contributions!(
  cache,
  info::RBInfo,
  op::NonlinearOperator,
  rbjac::RBMatAlgebraicContribution,
  rbspace_row::RBSpace,
  rbspace_col::RBSpace;
  kwargs...)

  mdeim_cache,rb_cache = cache
  st_mdeim = info.st_mdeim
  k = RBMatContributionMap()
  if isempty(rbjac)
    zero_rb_contribution(k,info,rbspace_row,rbspace_col)
  else
    collect_cache,coeff_cache = mdeim_cache
    jac = collect_reduced_jacobians!(collect_cache,op,rbjac;kwargs...)
    rb_jac_contribs = map(eachindex(rbjac)) do i
      coeff = rb_coefficient!(coeff_cache,rbjac[i],jac[i];st_mdeim)
      rb_contribution!(rb_cache,k,rbjac[i],coeff)
    end
    sum(rb_jac_contribs)
  end
end

function collect_reduced_residuals!(
  cache,op::NonlinearOperator,
  rbres::RBVecAlgebraicContribution)
  collect_reduced_residuals!(cache,op,rbres.affine_decompositions)
end

function collect_reduced_jacobians!(
  cache,
  op::NonlinearOperator,
  rbjac::RBMatAlgebraicContribution;
  kwargs...)
  collect_reduced_jacobians!(cache,op,rbjac.affine_decompositions;kwargs...)
end
