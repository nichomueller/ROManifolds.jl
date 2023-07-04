struct RBAlgebraicContribution
  dict::IdDict{Triangulation,RBAffineDecompositions}
end

RBAlgebraicContribution() = RBAlgebraicContribution(IdDict{Triangulation,RBAffineDecompositions}())

num_domains(a::RBAlgebraicContribution) = length(a.dict)

get_domains(a::RBAlgebraicContribution) = keys(a.dict)

function get_contribution(
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

function add_contribution!(
  a::RBAlgebraicContribution,
  trian::Triangulation,
  b::RBAffineDecompositions,
  op=+)

  if haskey(a.dict,trian)
    a.dict[trian] = op(a.dict[trian],b)
  else
    if op == +
      a.dict[trian] = b
    else
      a.dict[trian] = op(b)
    end
  end
  a
end

function get_measures(a::RBAlgebraicContribution)
  measures = Measures[]
  for (_,rbad) in a.dict
    push!(measures,rbad.integration_domain.meas)
  end
  unique(measures)
end

function assemble_rb_residual(
  c::RBAlgebraicContribution,
  sol::AbstractArray,
  μ::AbstractArray)

  m = get_measures(c)
  res = []
  for trian in get_domains(c)
    res_ad = get_contribution(c,trian)
    res_basis = get_basis(res_ad)
  end
end

function assemble_vector(
  feop::ParamTransientFEOperator,
  c::RBAlgebraicContribution,
  sols::AbstractArray,
  μ::AbstractArray)

  m = get_measures(c)
  vec = allocate_vector(c)
  for (trian,rbres) in c.dict
    vecdatum = _vecdata_residual(feop,solver,sols,μ,trian;m)
    idx_space = rbres.integration_domain.idx_space
    times = rbres.integration_domain.times
    @inbounds for (n,tn) in enumerate(times)
      vn = assemble_vector(vecdatum(μ,tn),feop.test)
      copyto!(view(vec,:,n),vn[idx_space])
    end
  end
  vec
end

function assemble_matrix(
  feop::ParamTransientFEOperator,
  c::RBAlgebraicContribution,
  sols::AbstractArray,
  μ::AbstractArray)

  m = get_measures(c)
  mats = Matrix{Float}[]
  for (trian,rbjac) in c.dict
    matdatum = _matdata_jacobian(feop,solver,sols,μ,trian;m)
    idx_space = rbjac.integration_domain.idx_space
    times = rbjac.integration_domain.times
    mat = zeros(length(idx_space),length(times))
    @inbounds for (n,tn) in enumerate(times)
      mn = assemble_matrix(matdatum(μ,tn),feop.trial(μ,tn),feop.test)
      copyto!(view(mat,:,n),reshape(mn,:,1)[idx_space])
    end
    push!(mats,mat)
  end
  mats
end
