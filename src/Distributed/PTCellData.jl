struct DistributedPTCellField{A,B} <: DistributedCellDatum
  fields::A
  metadata::B
  function DistributedPTCellField(
    fields::AbstractArray{<:PTCellField},
    metadata=nothing)

    A = typeof(fields)
    B = typeof(metadata)
    new{A,B}(fields,metadata)
  end
end

local_views(a::DistributedPTCellField) = a.fields

function CellData.CellField(f::AbstractPTFunction,trian::DistributedTriangulation)
  fields = map(trian.trians) do t
    PTCellField(f,t)
  end
  DistributedPTCellField(fields)
end

function (f::DistributedPTCellField)(x::DistributedCellPoint)
  evaluate!(nothing,f,x)
end

function Arrays.evaluate!(cache,f::DistributedPTCellField,x::DistributedCellPoint)
  map(f.fields,x.points) do f,x
    evaluate!(nothing,f,x)
  end
end

function Arrays.evaluate!(cache,k::Operation,a::DistributedPTCellField)
  fields = map(a.fields) do f
    evaluate!(nothing,k,f)
  end
  DistributedPTCellField(fields)
end

function Arrays.evaluate!(
  cache,k::Operation,a::DistributedPTCellField,b::DistributedPTCellField)
  fields = map(a.fields,b.fields) do f,g
    evaluate!(nothing,k,f,g)
  end
  DistributedPTCellField(fields)
end

function Arrays.evaluate!(cache,k::Operation,a::DistributedPTCellField,b::Number)
  fields = map(a.fields) do f
    evaluate!(nothing,k,f,b)
  end
  DistributedPTCellField(fields)
end

function Arrays.evaluate!(cache,k::Operation,b::Number,a::DistributedPTCellField)
  fields = map(a.fields) do f
    evaluate!(nothing,k,b,f)
  end
  DistributedPTCellField(fields)
end

function Arrays.evaluate!(cache,k::Operation,a::DistributedPTCellField,b::Function)
  fields = map(a.fields) do f
    evaluate!(nothing,k,f,b)
  end
  DistributedPTCellField(fields)
end

function Arrays.evaluate!(cache,k::Operation,b::Function,a::DistributedPTCellField)
  fields = map(a.fields) do f
    evaluate!(nothing,k,b,f)
  end
  DistributedPTCellField(fields)
end

function Arrays.evaluate!(cache,k::Operation,a::DistributedPTCellField...)
  fields = map(map(i->i.fields,a)...) do f...
    evaluate!(nothing,k,f...)
  end
  DistributedPTCellField(fields)
end

# Composition

Base.:(∘)(f::Function,g::DistributedPTCellField) = Operation(f)(g)
Base.:(∘)(f::Function,g::Tuple{DistributedPTCellField,DistributedPTCellField}) = Operation(f)(g[1],g[2])
Base.:(∘)(f::Function,g::Tuple{DistributedPTCellField,Number}) = Operation(f)(g[1],g[2])
Base.:(∘)(f::Function,g::Tuple{Number,DistributedPTCellField}) = Operation(f)(g[1],g[2])
Base.:(∘)(f::Function,g::Tuple{DistributedPTCellField,Function}) = Operation(f)(g[1],g[2])
Base.:(∘)(f::Function,g::Tuple{Function,DistributedPTCellField}) = Operation(f)(g[1],g[2])
Base.:(∘)(f::Function,g::Tuple{Vararg{DistributedPTCellField}}) = Operation(f)(g...)

# Unary ops

for op in (:symmetric_part,:inv,:det,:abs,:abs2,:+,:-,:tr,:transpose,:adjoint,:grad2curl,:real,:imag,:conj)
  @eval begin
    ($op)(a::DistributedPTCellField) = Operation($op)(a)
  end
end

# Binary ops

for op in (:inner,:outer,:double_contraction,:+,:-,:*,:cross,:dot,:/)
  @eval begin
    ($op)(a::DistributedPTCellField,b::DistributedPTCellField) = Operation($op)(a,b)
    ($op)(a::DistributedPTCellField,b::Number) = Operation($op)(a,b)
    ($op)(a::Number,b::DistributedPTCellField) = Operation($op)(a,b)
    ($op)(a::DistributedPTCellField,b::Function) = Operation($op)(a,b)
    ($op)(a::Function,b::DistributedPTCellField) = Operation($op)(a,b)
  end
end

Base.broadcasted(f,a::DistributedPTCellField,b::DistributedPTCellField) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(f,a::Number,b::DistributedPTCellField) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(f,a::DistributedPTCellField,b::Number) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(f,a::Function,b::DistributedPTCellField) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(f,a::DistributedPTCellField,b::Function) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(::typeof(*),::typeof(∇),f::DistributedPTCellField) = Operation(Fields._extract_grad_diag)(∇(f))
Base.broadcasted(::typeof(*),s::Fields.ShiftedNabla,f::DistributedPTCellField) = Operation(Fields._extract_grad_diag)(s(f))

dot(::typeof(∇),f::DistributedPTCellField) = divergence(f)
outer(::typeof(∇),f::DistributedPTCellField) = gradient(f)
outer(f::DistributedPTCellField,::typeof(∇)) = transpose(gradient(f))
cross(::typeof(∇),f::DistributedPTCellField) = curl(f)

# Differential ops

function Fields.gradient(a::DistributedPTCellField)
  DistributedPTCellField(map(gradient,a.fields))
end

function Fields.divergence(a::DistributedPTCellField)
  DistributedPTCellField(map(divergence,a.fields))
end

function Fields.DIV(a::DistributedPTCellField)
  DistributedPTCellField(map(DIV,a.fields))
end

function Fields.∇∇(a::DistributedPTCellField)
  DistributedPTCellField(map(∇∇,a.fields))
end

function Fields.curl(a::DistributedPTCellField)
  DistributedPTCellField(map(curl,a.fields))
end

# Integration related

struct DistributedReducedMeasure{A<:AbstractArray{<:ReducedMeasure}} <: GridapType
  measures::A
end

local_views(a::DistributedReducedMeasure) = a.measures

function ReducedMeasure(t::DistributedTriangulation,args...)
  measures = map(t.trians) do trian
    ReducedMeasure(trian,args...)
  end
  DistributedReducedMeasure(measures)
end

function CellData.get_cell_points(a::DistributedReducedMeasure)
  DistributedCellPoint(map(get_cell_points,a.measures))
end

struct DistributedPTDomainContribution{A<:AbstractArray{<:PTDomainContribution}} <: GridapType
  contribs::A
end

local_views(a::DistributedPTDomainContribution) = a.contribs

function Base.getindex(c::DistributedPTDomainContribution,t::DistributedTriangulation)
  map(getindex,c.contribs,t.trians)
end

function Fields.integrate(f::DistributedPTCellField,b::DistributedMeasure)
  contribs = map(f.fields,b.measures) do f,m
    integrate(f,m)
  end
  DistributedPTDomainContribution(contribs)
end

function Fields.integrate(f::Function,b::DistributedMeasure)
  contribs = map(b.measures) do m
    integrate(f,m)
  end
  DistributedPTDomainContribution(contribs)
end

function Fields.integrate(f::Number,b::DistributedMeasure)
  contribs = map(b.measures) do m
    integrate(f,m)
  end
  DistributedPTDomainContribution(contribs)
end

function Base.sum(a::DistributedPTDomainContribution)
  sum(map(sum,a.contribs))
end

function (+)(a::DistributedPTDomainContribution,b::DistributedPTDomainContribution)
  contribs = map(+,a.contribs,b.contribs)
  DistributedPTDomainContribution(contribs)
end

function (-)(a::DistributedPTDomainContribution,b::DistributedPTDomainContribution)
  contribs = map(-,a.contribs,b.contribs)
  DistributedPTDomainContribution(contribs)
end

function (*)(a::Number,b::DistributedPTDomainContribution)
  contribs = map(b.contribs) do b
    a*b
  end
  DistributedPTDomainContribution(contribs)
end

(*)(a::DistributedPTDomainContribution,b::Number) = b*a
