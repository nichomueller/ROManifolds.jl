function Base.iterate(
  f::DistributedCellField{<:Vector{<:SingleFieldPTFEFunction}}
  )

  fit,nit = map(local_views(f)) do f
    first(iterate(f))
  end |> tuple_of_arrays
  return (fit,first(nit)),first(nit)
end

function Base.iterate(
  f::DistributedCellField{<:Vector{<:SingleFieldPTFEFunction}},
  state)

  fn = map(local_views(f)) do f
    iterate(f,state)
  end
  if isa(fn,AbstractVector{Nothing})
    return nothing
  else
    fit,nit = tuple_of_arrays(fn)
    return (first.(fit),first(nit)),first(nit)
  end
end

function FESpaces.get_triangulation(meas::DistributedMeasure)
  @assert false
  trian,model = map(meas.measures) do m
    trian = m.quad.trian
    model = get_background_model(trian)
    trian,model
  end |> tuple_of_arrays
  model = DistributedDiscreteModel(model)
  DistributedTriangulation(trian,model)
end

function Fields.integrate(
  a::CollectionPTIntegrand{<:PTIntegrand{<:DistributedCellField},N}) where N

  cont = Vector{DistributedDomainContribution}(undef,N)
  for i = 1:N
    op,integrand = a[i]
    imeas = integrand.meas
    conti = map(integrand.object.fields,imeas.measures) do f,m
      integral = integrate(f,m.quad)
      lazy_map(Broadcasting(op),integral)
    end
    cont[i] = conti
  end
  sum(cont)
end

Base.broadcasted(f,a::Nothing,b::DistributedCellField) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(f,a::DistributedCellField,b::Nothing) = Operation((i,j)->f.(i,j))(a,b)

(+)(::Nothing,b::DistributedDomainContribution) = b
(+)(a::DistributedDomainContribution,::Nothing) = a
(-)(a::DistributedDomainContribution,::Nothing) = a

function (-)(::Nothing,b::DistributedDomainContribution)
  contribs = map(local_views(b)) do bi
    for (trian,array) in bi.dict
      bi.dict[trian] = lazy_map(Broadcasting(-),array)
    end
    bi
  end
  DistributedDomainContribution(contribs)
end
