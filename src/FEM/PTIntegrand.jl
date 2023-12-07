struct PTIntegrand{T,M}
  object::T
  meas::M
end

const ∫ₚ = PTIntegrand

function Fields.integrate(a::PTIntegrand)
  integrate(a.object,a.meas)
end

struct CollectionPTIntegrand{I,N}
  operations::NTuple{N,Union{typeof(+),typeof(-)}}
  integrands::NTuple{N,I}
end

function Base.getindex(a::CollectionPTIntegrand,i::Int)
  a.operations[i],a.integrands[i]
end

for op in (:+,:-)
  @eval begin
    function ($op)(a::PTIntegrand,b::PTIntegrand)
      CollectionPTIntegrand((+,$op),(a,b))
    end

    function ($op)(a::CollectionPTIntegrand,b::PTIntegrand)
      CollectionPTIntegrand((a.operations...,$op),(a.integrands...,b))
    end

    function ($op)(a::PTIntegrand,b::CollectionPTIntegrand)
      CollectionPTIntegrand(($op,b.operations...),(a,b.integrands...))
    end

    function ($op)(a::CollectionPTIntegrand,b::CollectionPTIntegrand)
      operations = (a.operations...,b.operations...)
      integrands = (a.integrands...,b.integrands...)
      CollectionPTIntegrand(operations,integrands)
    end
  end
end

function Fields.integrate(a::CollectionPTIntegrand{I,N} where I) where N
  cont = DomainContribution()
  for i = 1:N
    op,integrand = a[i]
    imeas = integrand.meas
    itrian = get_triangulation(imeas)
    integral = integrate(integrand.object,imeas.quad)
    add_contribution!(cont,itrian,integral,op)
  end
  cont
end

# Interface that allows to entirely eliminate terms from the (PT)DomainContribution
for op in (:inner,:outer,:double_contraction,:+,:-,:*,:cross,:dot,:/)
  @eval begin
    ($op)(::Nothing,::Nothing) = nothing
    ($op)(::Any,::Nothing) = nothing
    ($op)(::Nothing,::Any) = nothing
  end
end

Base.adjoint(::Nothing) = nothing
Base.broadcasted(f,a::Nothing,b::Nothing) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(f,a::Nothing,b::CellField) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(f,a::CellField,b::Nothing) = Operation((i,j)->f.(i,j))(a,b)
LinearAlgebra.dot(::typeof(∇),::Nothing) = nothing

Fields.gradient(::Nothing) = nothing

∂ₚt(::Nothing) = nothing

CellData.integrate(::Nothing,args...) = nothing

CellData.integrate(::Any,::Nothing) = nothing

PTIntegrand(::Nothing,::Measure) = nothing

for T in (:DomainContribution,:PTIntegrand)#,:CollectionPTIntegrand)
  @eval begin
    (+)(::Nothing,b::$T) = b
    (+)(a::$T,::Nothing) = a
    (-)(a::$T,::Nothing) = a
  end
end

function (-)(::Nothing,b::DomainContribution)
  for (trian,array) in b.dict
    b.dict[trian] = lazy_map(Broadcasting(-),array)
  end
  b
end

function (-)(::Nothing,b::PTIntegrand)
  PTIntegrand(-b.object,b.meas)
end

# function (-)(::Nothing,b::CollectionPTIntegrand)
#   _neg_sign(::typeof(+)) = -
#   _neg_sign(::typeof(-)) = +
#   CollectionPTIntegrand(map(_neg_sign,b.operations),b.integrands)
# end

function FESpaces.collect_cell_vector(::FESpace,::Nothing,args...)
  nothing
end

function FESpaces.collect_cell_matrix(::FESpace,::FESpace,::Nothing,args...)
  nothing
end
