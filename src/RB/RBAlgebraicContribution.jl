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

function Base.show(io::IO,a::RBAlgebraicContribution)
  printstyled("RB ALGEBRAIC CONTRIBUTIONS INFO\n";underline=true)
  for trian in get_domains(a)
    atrian = a[trian]
    red_method = get_reduction_method(atrian)
    red_var = get_reduced_variable(atrian)
    nbs = length(atrian.basis_space)
    nbt = size(atrian.basis_time[1],2)
    print(io,"$red_var on a $trian, reduction in $red_method\n")
    print(io,"number basis vectors in (space, time) = ($nbs,$nbt)\n")
  end
end

function Base.show(io::IO,a::Vector{<:RBAlgebraicContribution})
  map(x->show(io,x),a)
end

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
  Tpath = joinpath(path,"type")
  save(Tpath,T)
  for (i,trian) in enumerate(get_domains(a))
    ai = a[trian]
    save(cpath*"_$i",ai)
    save(tpath*"_$i",trian)
  end
end

for (AC,AD) in zip((:RBVecAlgebraicContribution,:RBMatAlgebraicContribution),
                   (:RBVecAffineDecomposition,:RBMatAffineDecomposition))
  @eval begin
    function load_algebraic_contrib(path::String,::Type{$AC})
      cpath = joinpath(path,"contrib")
      tpath = joinpath(path,"trian")
      T = load(joinpath(path,"type"),DataType)
      a = $AC(T)
      i = 1
      while isfile(correct_path(cpath*"_$i"))
        ai = load(cpath*"_$i",$AD{T})
        ti = load(tpath*"_$i",Triangulation)
        add_contribution!(a,ti,ai)
        i += 1
      end
      a
    end
  end
end

function save(info::RBInfo,a::RBVecAlgebraicContribution)
  path = joinpath(info.rb_path,"rb_rhs")
  save_algebraic_contrib(path,a)
end

function load(info::RBInfo,T::Type{RBVecAlgebraicContribution})
  path = joinpath(info.rb_path,"rb_rhs")
  load_algebraic_contrib(path,T)
end

function save(info::RBInfo,a::Vector{<:RBMatAlgebraicContribution})
  for i = eachindex(a)
    path = joinpath(info.rb_path,"rb_lhs_$i")
    save_algebraic_contrib(path,a[i])
  end
end

function load(info::RBInfo,::Type{Vector{RBMatAlgebraicContribution}})
  T = load(joinpath(joinpath(info.rb_path,"rb_lhs_1"),"type"),DataType)
  njacs = num_active_dirs(info.rb_path)
  ad_jacs = Vector{RBMatAlgebraicContribution{T}}(undef,njacs)
  for i = 1:njacs
    path = joinpath(info.rb_path,"rb_lhs_$i")
    ad_jacs[i] = load_algebraic_contrib(path,RBMatAlgebraicContribution)
  end
  ad_jacs
end

function collect_compress_rhs_lhs(
  info::RBInfo,
  feop::PTFEOperator{Affine},
  fesolver::PThetaMethod,
  rbspace,
  snaps,
  μ::Table;
  nsnaps_mdeim=20)

  snapsθ = recenter(fesolver,snaps,μ)
  _snapsθ,_μ = snapsθ[1:nsnaps_mdeim],μ[1:nsnaps_mdeim]
  rhs = collect_compress_rhs(info,feop,fesolver,rbspace,_snapsθ,_μ)
  lhs = collect_compress_lhs(info,feop,fesolver,rbspace,_snapsθ,_μ)
  show(rhs),show(lhs)
  rhs,lhs
end

function collect_compress_rhs_lhs(
  info::RBInfo,
  feop::PTFEOperator{Nonlinear},
  fesolver::PThetaMethod,
  rbspace,
  snaps::PTArray,
  μ::Table;
  nsnaps_mdeim=20)

  snapsθ = recenter(fesolver,snaps,μ)
  _snapsθ,_μ = snapsθ[1:nsnaps_mdeim],μ[1:nsnaps_mdeim]

  lfeop = linear_operator(feop)
  rhs = collect_compress_rhs(info,lfeop,fesolver,rbspace,_snapsθ,_μ)
  lhs = collect_compress_lhs(info,lfeop,fesolver,rbspace,_snapsθ,_μ)
  nlfeop = nonlinear_operator(feop)
  nllhs = collect_compress_lhs(info,nlfeop,fesolver,rbspace,_snapsθ,_μ)
  show(rhs),show(lhs)

  rhs,lhs,nllhs
end

function collect_compress_rhs(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbspace::RBSpace{T},
  snaps::PTArray,
  μ::Table) where T

  times = get_times(fesolver)
  ress,trian = collect_residuals_for_trian(fesolver,feop,snaps,μ,times)
  ad_res = RBVecAlgebraicContribution(T)
  compress_component!(ad_res,info,feop,ress,trian,times,rbspace)
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
  ad_jacs = Vector{RBMatAlgebraicContribution{T}}(undef,njacs)
  for i = 1:njacs
    combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*x-θ*y
    jacs,trian = collect_jacobians_for_trian(fesolver,feop,snaps,μ,times;i)
    ad_jac_i = RBMatAlgebraicContribution(T)
    compress_component!(ad_jac_i,info,feop,jacs,trian,times,rbspace,rbspace;combine_projections)
    ad_jacs[i] = ad_jac_i
  end
  return ad_jacs
end

function compress_component!(
  contrib::RBAlgebraicContribution,
  info::RBInfo,
  feop::PTFEOperator,
  snaps::Vector{<:NnzMatrix},
  trian::Base.KeySet{Triangulation},
  args...;
  kwargs...)

  for (i,ti) in enumerate(trian)
    si = snaps[i]
    if !iszero(si)
      ci = RBAffineDecomposition(info,feop,si,ti,args...;kwargs...)
      add_contribution!(contrib,ti,ci)
    end
  end
end

function collect_rhs_contributions!(
  cache,
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbres::RBVecAlgebraicContribution{T},
  rbspace::RBSpace{T},
  sols::PTArray,
  params::Table) where T

  coeff_cache,rb_cache = cache
  st_mdeim = info.st_mdeim
  k = RBVecContributionMap(T)
  rb_res_contribs = Vector{PTArray{Vector{T}}}(undef,num_domains(rbres))
  if iszero(rbres)
    nrow = get_rb_ndofs(rbspace)
    contrib = PTArray([zeros(T,nrow) for _ = eachindex(params)])
    rb_res_contribs[i] = contrib
  else
    for (i,t) in enumerate(get_domains(rbres))
      rbrest = rbres[t]
      coeff = rhs_coefficient!(coeff_cache,feop,fesolver,rbrest,sols,params;st_mdeim)
      rb_res_contribs[i] = rb_contribution!(rb_cache,k,rbrest,coeff)
    end
  end
  return sum(rb_res_contribs)
end

function collect_lhs_contributions!(
  cache,
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbjacs::Vector{RBMatAlgebraicContribution{T}},
  rbspace::RBSpace{T},
  args...) where T

  njacs = length(rbjacs)
  rb_jacs_contribs = Vector{PTArray{Matrix{T}}}(undef,njacs)
  for i = 1:njacs
    rb_jac_i = rbjacs[i]
    rb_jacs_contribs[i] = collect_lhs_contributions!(
      cache,info,feop,fesolver,rb_jac_i,rbspace,rbspace,args...;i)
  end
  return sum(rb_jacs_contribs)
end

function collect_lhs_contributions!(
  cache,
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbjac::RBMatAlgebraicContribution{T},
  rbspace_row::RBSpace{T},
  rbspace_col::RBSpace{T},
  sols::PTArray,
  params::Table;
  kwargs...) where T

  coeff_cache,rb_cache = cache
  trian = get_domains(rbjac)
  st_mdeim = info.st_mdeim
  k = RBMatContributionMap(T)
  rb_jac_contribs = Vector{PTArray{Matrix{T}}}(undef,num_domains(rbjac))
  if iszero(rbjac)
    nrow = get_rb_ndofs(rbspace_row)
    ncol = get_rb_ndofs(rbspace_col)
    contrib = PTArray([zeros(T,nrow,ncol) for _ = eachindex(params)])
    rb_jac_contribs[i] = contrib
  else
    for (i,t) in enumerate(trian)
      rbjact = rbjac[t]
      coeff = lhs_coefficient!(coeff_cache,feop,fesolver,rbjact,sols,params;st_mdeim,kwargs...)
      rb_jac_contribs[i] = rb_contribution!(rb_cache,k,rbjact,coeff)
    end
  end
  return sum(rb_jac_contribs)
end
