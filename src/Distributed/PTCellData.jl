function FESpaces.get_triangulation(meas::DistributedMeasure)
  trian = map(local_views(meas)) do m
    m.quad.trian
  end
  DistributedTriangulation(trian)
end

function Fields.integrate(a::CollectionPTIntegrand{N,<:PTIntegrand{<:DistributedCellField}}) where N
  cont = Vector{DistributedDomainContribution}(undef,N)
  for i = 1:N
    op,integrand = a[i]
    imeas = integrand.meas
    conti = map(local_views(imeas)) do m
      integral = integrate(integrand.object,m.quad)
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
