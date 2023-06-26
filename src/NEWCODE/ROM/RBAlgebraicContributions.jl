struct RBAlgebraicContribution
  dict::IdDict{Triangulation,RBAffineDecompositions}
end

RBAlgebraicContribution() = RBAlgebraicContribution(IdDict{Triangulation,RBAffineDecompositions}())

Gridap.CellData.num_domains(a::RBAlgebraicContribution) = length(a.dict)

Gridap.CellData.get_domains(a::RBAlgebraicContribution) = keys(a.dict)

function Gridap.CellData.get_contribution(
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

function Gridap.CellData.add_contribution!(
  a::DomainContribution,
  trian::Triangulation,
  b::RBAffineDecompositions,
  op=+)

  if haskey(a.dict,trian)
    @notimplemented
    # a.dict[trian] = lazy_map(Broadcasting(op),a.dict[trian],b)
  else
    if op == +
     a.dict[trian] = b
    else
      @notimplemented
    #  a.dict[trian] = lazy_map(Broadcasting(op),b)
    end
  end
  a
end

function get_measures(a::RBAlgebraicContribution)
  measures = Measures[]
  for (_,rbad) in a.dict
    push!(measures,rbad.integration_domain.meas)
  end
  measures
end

function Gridap.FESpaces.assemble_vector(
  feop::ParamTransientFEOperator,
  c::RBAlgebraicContribution,
  sol::AbstractArray,
  μ::AbstractArray)

  m = get_measures(c)
  vec = allocate_vector(c)
  for (trian,rbad) in c.dict
    vecdatum = _vecdata_residual(feop,solver,sol,μ,trian;m)
    idx_space = rbad.integration_domain.idx_space
    times = rbad.integration_domain.times
    @inbounds for (n,tn) in enumerate(times)
      vn = assemble_vector(vecdatum(μ,tn),feop.test)
      copyto!(view(vec,:,n),vn[idx_space])
    end
  end
  vec
end

function Gridap.FESpaces.assemble_matrix(
  feop::ParamTransientFEOperator,
  c::RBAlgebraicContribution,
  sol::AbstractArray,
  μ::AbstractArray)

  m = get_measures(c)
  mat = allocate_matrix(c)
  for (trian,rbad) in c.dict
    matdatum = _matdata_jacobian(feop,solver,sol,μ,trian;m)
    idx_space = rbad.integration_domain.idx_space
    times = rbad.integration_domain.times
    @inbounds for (n,tn) in enumerate(times)
      mn = assemble_matrix(matdatum(μ,tn),feop.trial(μ,tn),feop.test)
      copyto!(view(mat,:,n),reshape(mn,:,1)[idx_space])
    end
  end
  mat
end
