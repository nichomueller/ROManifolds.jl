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
Base.eltype(::AbstractRBAlgebraicContribution{T}) where T = T

function CellData.add_contribution!(
  a::AbstractRBAlgebraicContribution,
  trian::Triangulation,
  b)

  @check !haskey(a.dict,trian)
  a.dict[trian] = b
  a
end

function save_algebraic_contrib(path::String,a::AbstractRBAlgebraicContribution{T}) where T
  create_dir!(path)
  cpath = joinpath(path,"contrib")
  tpath = joinpath(path,"trian")
  Tpath = joinpath(path,"type")
  save(Tpath,T)
  for (i,trian) in enumerate(get_domains(a))
    ai = a[trian]
    save(cpath*"_$i",ai)
    save(tpath*"_$i",trian)
  end
end

function load_algebraic_contrib(path::String,::Type{AbstractRBAlgebraicContribution})
  cpath = joinpath(path,"contrib")
  tpath = joinpath(path,"trian")
  T = load(joinpath(path,"type"),DataType)
  a = RBAlgebraicContribution(T)
  i = 1
  while isfile(correct_path(cpath*"_$i"))
    ai = load(cpath*"_$i",RBAffineDecomposition{T})
    ti = load(tpath*"_$i",Triangulation)
    add_contribution!(a,ti,ai)
    i += 1
  end
  a
end

function save(info::RBInfo,a::AbstractRBAlgebraicContribution{T}) where T
  if info.save_structures
    path = joinpath(info.rb_path,"rb_rhs")
    save_algebraic_contrib(path,a)
  end
end

function load(info::RBInfo,T::Type{AbstractRBAlgebraicContribution})
  path = joinpath(info.rb_path,"rb_rhs")
  load_algebraic_contrib(path,T)
end

function save(info::RBInfo,a::Vector{<:AbstractRBAlgebraicContribution{T}}) where T
  if info.save_structures
    for i = eachindex(a)
      path = joinpath(info.rb_path,"rb_lhs_$i")
      save_algebraic_contrib(path,a[i])
    end
  end
end

function load(info::RBInfo,::Type{Vector{AbstractRBAlgebraicContribution}})
  ad_jac1 = load_algebraic_contrib(joinpath(info.rb_path,"rb_lhs_1"),AbstractRBAlgebraicContribution)
  T = eltype(ad_jac1)
  ad_jacs = RBAlgebraicContribution{T}[]
  push!(ad_jacs,ad_jac1)
  i = 2
  while isfile(correct_path(joinpath(info.rb_path,"rb_lhs_$i")))
    path = joinpath(info.rb_path,"rb_lhs_$i")
    push!(ad_jacs,load_algebraic_contrib(path,AbstractRBAlgebraicContribution))
    i += 1
  end
  ad_jacs
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

  rb_res_contribs = Vector{PTArray{Vector{T}}}(undef,num_domains(rbres))
  for (i,t) in enumerate(trian)
    rbrest = rbres[t]
    coeff = rhs_coefficient!(coeff_cache,feop,fesolver,rbrest,args...;st_mdeim)
    rb_res_contribs[i] = rb_contribution!(rb_cache,rbrest,coeff)
  end
  return sum(rb_res_contribs)
end

function collect_lhs_contributions!(
  cache,
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbjacs::Vector{<:AbstractRBAlgebraicContribution{T}},
  args...) where T

  njacs = length(rbjacs)
  rb_jacs_contribs = Vector{PTArray{Matrix{T}}}(undef,njacs)
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
  for (i,t) in enumerate(trian)
    rbjact = rbjac[t]
    coeff = lhs_coefficient!(coeff_cache,feop,fesolver,rbjact,args...;st_mdeim,kwargs...)
    rb_jac_contribs[i] = rb_contribution!(rb_cache,rbjact,coeff)
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
