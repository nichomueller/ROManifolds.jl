abstract type AbstractRBAlgebraicContribution{T} end

CellData.num_domains(a::AbstractRBAlgebraicContribution) = length(a.dict)
CellData.get_domains(a::AbstractRBAlgebraicContribution) = keys(a.dict)

function CellData.get_contribution(
  a::AbstractRBAlgebraicContribution,
  trian::Triangulation)

  if haskey(a.dict,trian)
    return a.dict[trian]
  else
    @unreachable """\n
    There is not contribution associated with the given mesh in this RBAlgebraicContribution object.
    """
  end
end

Base.getindex(a::AbstractRBAlgebraicContribution,trian::Triangulation) = get_contribution(a,trian)

function CellData.add_contribution!(
  a::AbstractRBAlgebraicContribution,
  trian::Triangulation,
  b)

  @check !haskey(a.dict,trian)
  a.dict[trian] = b
  a
end

function save(info::RBInfo,a::AbstractRBAlgebraicContribution)
  if info.save_structures
    path = joinpath(info.rb_path,"rb_rhs")
    save(path,a)
  end
end

function load(info::RBInfo,T::Type{AbstractRBAlgebraicContribution})
  path = joinpath(info.rb_path,"rb_rhs")
  load(path,T)
end

function save(info::RBInfo,a::Vector{<:AbstractRBAlgebraicContribution})
  if info.save_structures
    path = joinpath(info.rb_path,"rb_lhs")
    save(path,a)
  end
end

function load(info::RBInfo,T::Type{Vector{AbstractRBAlgebraicContribution}})
  path = joinpath(info.rb_path,"rb_lhs")
  load(path,T)
end

struct RBAlgebraicContribution{T} <: AbstractRBAlgebraicContribution{T}
  dict::IdDict{Triangulation,RBAffineDecomposition{T}}
  function RBAlgebraicContribution(::Type{T}) where T
    new{T}(IdDict{Triangulation,RBAffineDecomposition{T}}())
  end
end

function collect_rhs_contributions!(
  cache,
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbres::RBAlgebraicContribution{T},
  args...) where T

  coeff_cache,rb_cache = cache
  trian = get_domains(rbres)
  st_mdeim = info.st_mdeim

  rb_res_contribs = Vector{PTArray{Matrix{T}}}(undef,num_domains(rbres))
  for t in trian
    rbrest = rbres[t]
    coeff = rhs_coefficient!(coeff_cache,feop,fesolver,rbrest,trian,args...;st_mdeim)
    contrib = rb_contribution!(rb_cache,rbrest,coeff)
    push!(rb_res_contribs,contrib)
  end
  return sum(rb_res_contribs)
end

function collect_lhs_contributions!(
  cache,
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbjacs::Vector{AbstractRBAlgebraicContribution{T}},
  args...) where T

  njacs = length(rbjacs)
  rb_jacs_contribs = Vector{<:PTArray{Matrix{T}}}(undef,njacs)
  for i = 1:njacs
    rb_jac_i = rbjacs[i]
    rb_jacs_contribs[i] = collect_lhs_contributions!(cache,info,feop,fesolver,rb_jac_i,args...;i)
  end
  return sum(rb_jacs_contribs)
end

function collect_lhs_contributions!(
  cache,
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbjac::RBAlgebraicContribution{T},
  args...;
  kwargs...) where T

  coeff_cache,rb_cache = cache
  trian = get_domains(rbjac)
  st_mdeim = info.st_mdeim

  rb_jac_contribs = Vector{PTArray{Matrix{T}}}(undef,num_domains(rbjac))
  for t in 1:trian
    rbjact = rbjac[t]
    coeff = lhs_coefficient!(coeff_cache,feop,fesolver,rbjact,trian,args...;st_mdeim,kwargs...)
    contrib = rb_contribution!(rb_cache,rbjact,coeff)
    push!(rb_jac_contribs,contrib)
  end
  return sum(rb_jac_contribs)
end

struct RBBlockAlgebraicContribution{T} <: AbstractRBAlgebraicContribution{T}
  block::Matrix{RBAlgebraicContribution{T}}
  touched::Vector{Int}

  function RBBlockAlgebraicContribution(
    block::Matrix{RBAlgebraicContribution{T}},
    touched::Vector{Int}) where T

    new{T}(block,touched)
  end
end

function Arrays.testvalue(
  ::Type{RBBlockAlgebraicContribution{T}},
  feop::PTFEOperator,
  size::Vararg{Int}) where T

  blocks = Matrix{RBAlgebraicContribution{T}}(undef,size)
  touched = Matrix{Bool}(undef,size)
  RBBlockAffineDecomposition(blocks,touched)
end
