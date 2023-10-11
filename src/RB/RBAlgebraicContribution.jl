struct RBAlgebraicContribution{T}
  dict::IdDict{Triangulation,RBAffineDecomposition{T}}
  function RBAlgebraicContribution(::Type{T}) where T
    new{T}(IdDict{Triangulation,RBAffineDecomposition{T}}())
  end
end

CellData.num_domains(a::RBAlgebraicContribution) = length(a.dict)
CellData.get_domains(a::RBAlgebraicContribution) = keys(a.dict)

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
Base.eltype(::RBAlgebraicContribution{T}) where T = T

function Arrays.testvalue(
  ::Type{RBAlgebraicContribution{T}},
  feop::PTFEOperator;
  kwargs...) where T

  a = RBAlgebraicContribution(T)
  trian = get_triangulation(feop.test)
  ad = testvalue(RBAlgebraicContribution{T},feop;kwargs...)
  add_contribution!(a,trian,ad)
end

function CellData.add_contribution!(
  a::RBAlgebraicContribution,
  trian::Triangulation,
  b)

  @check !haskey(a.dict,trian)
  a.dict[trian] = b
  a
end

function save_algebraic_contrib(path::String,a::RBAlgebraicContribution{T}) where T
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

function load_algebraic_contrib(path::String,::Type{RBAlgebraicContribution})
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

function save(info::RBInfo,a::RBAlgebraicContribution{T}) where T
  if info.save_structures
    path = joinpath(info.rb_path,"rb_rhs")
    save_algebraic_contrib(path,a)
  end
end

function load(info::RBInfo,T::Type{RBAlgebraicContribution})
  path = joinpath(info.rb_path,"rb_rhs")
  load_algebraic_contrib(path,T)
end

function save(info::RBInfo,a::Vector{RBAlgebraicContribution{T}}) where T
  if info.save_structures
    for i = eachindex(a)
      path = joinpath(info.rb_path,"rb_lhs_$i")
      save_algebraic_contrib(path,a[i])
    end
  end
end

function load(info::RBInfo,::Type{Vector{RBAlgebraicContribution}})
  ad_jac1 = load_algebraic_contrib(joinpath(info.rb_path,"rb_lhs_1"),RBAlgebraicContribution)
  T = eltype(ad_jac1)
  ad_jacs = RBAlgebraicContribution{T}[]
  push!(ad_jacs,ad_jac1)
  i = 2
  while isdir(joinpath(info.rb_path,"rb_lhs_$i"))
    path = joinpath(info.rb_path,"rb_lhs_$i")
    push!(ad_jacs,load_algebraic_contrib(path,RBAlgebraicContribution))
    i += 1
  end
  ad_jacs
end

function collect_compress_rhs_lhs(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbspace::RBSpace,
  snaps::Snapshots,
  μ::Table)

  nsnaps = info.nsnaps_system
  snapsθ = recenter(fesolver,snaps,μ)
  _snapsθ,_μ = snapsθ[1:nsnaps],μ[1:nsnaps]
  rhs = collect_compress_rhs(info,feop,fesolver,rbspace,_snapsθ,_μ)
  lhs = collect_compress_lhs(info,feop,fesolver,rbspace,_snapsθ,_μ)
  rhs,lhs
end

function collect_compress_rhs(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbspace::RBSpace,
  snaps::PTArray,
  μ::Table)

  times = get_times(fesolver)
  ress,trian = collect_residuals_for_trian(fesolver,feop,snaps,μ,times)
  ad_res = compress_component(info,feop,ress,trian,times,rbspace)
  return ad_res
end

function collect_compress_lhs(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbspace::RBSpace{T},
  snaps::PTArray,
  μ::Table) where T

  times = get_times(fesolver)
  θ = fesolver.θ

  njacs = length(feop.jacs)
  ad_jacs = Vector{RBAlgebraicContribution{T}}(undef,njacs)
  for i = 1:njacs
    combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*x-θ*y
    jacs,trian = collect_jacobians_for_trian(fesolver,feop,snaps,μ,times;i)
    ad_jacs[i] = compress_component(info,feop,jacs,trian,times,rbspace,rbspace;combine_projections)
  end
  return ad_jacs
end

function compress_component(
  info::RBInfo,
  feop::PTFEOperator,
  snaps::Vector{NnzMatrix{T}},
  trian::Base.KeySet{Triangulation},
  args...;
  kwargs...) where T

  contrib = RBAlgebraicContribution(T)
  for (i,ti) in enumerate(trian)
    si = snaps[i]
    ci = RBAffineDecomposition(info,feop,si,ti,args...;kwargs...)
    add_contribution!(contrib,ti,ci)
  end
  contrib
end

function collect_rhs_contributions!(
  cache,
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbres::RBAlgebraicContribution{T},
  ::RBSpace{T},
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
  rbjacs::Vector{RBAlgebraicContribution{T}},
  ::RBSpace{T},
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
