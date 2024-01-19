# struct PField{F} <: Field
#   fields::AbstractVector{F}
# end

# const PFieldGradient = PField{FieldGradient{N,F}} where {N,F}
# const PGenericField = PField{GenericField{T}} where T
# const PZeroField = PField{ZeroField{F}} where F

# CellData.FieldGradient{N}(f::PField) where N = PField(FieldGradient{N}.(f.fields))
# CellData.FieldGradient{N}(f::AbstractVector{<:PField}) where N = PField(FieldGradient{N}.(f.fields))
# CellData.FieldGradient{N}(f::AbstractPFunction) where N = PField(get_fields(f,:FieldGradient;N))
# CellData.GenericField(f::AbstractPFunction) = PField(get_fields(f,:GenericField))
# CellData.ZeroField(f::AbstractPFunction) = PField(get_fields(f,:ZeroField))

# Fields.gradient(f::PField) = FieldGradient{1}(f)
# Fields.gradient(f::PFieldGradient{N}) where N = FieldGradient{N+1}(f.fields)

# Base.size(f::PField) = size(f.fields)
# Base.length(f::PField) = length(f.fields)
# Base.eachindex(f::PField) = eachindex(f.fields)
# Base.IndexStyle(::Type{<:PField}) = IndexLinear()
# Base.getindex(f::PFieldGradient{N,F} where F,i::Integer) where N = f.fields[i]
# Base.getindex(f::PGenericField,i::Integer) = f.fields[i]
# Base.getindex(f::PZeroField,i::Integer) = f.fields[i]
# Arrays.testitem(f::PField) = f[1]

# for T in (:Point,:(AbstractArray{<:Point}))
#   @eval begin
#     Arrays.testargs(f::PField,x::$T) = testargs(f[1],x)

#     function Arrays.return_cache(f::PField,x::$T)
#       fi = testitem(f)
#       li = return_cache(fi,x)
#       fix = evaluate!(li,fi,x)
#       l = Vector{typeof(li)}(undef,size(f.fields))
#       g = Vector{typeof(fix)}(undef,size(f.fields))
#       for i in eachindex(f.fields)
#         l[i] = return_cache(f.fields[i],x)
#       end
#       PArray(g),l
#     end

#     function Arrays.evaluate!(cache,f::PField,x::$T)
#       g,l = cache
#       for i in eachindex(f.fields)
#         g[i] = evaluate!(l[i],f.fields[i],x)
#       end
#       g
#     end
#   end
# end

# function Arrays.return_value(
#   b::LagrangianDofBasis,
#   field::OperationField{<:PField})

#   f1 = OperationField(testitem(field.op),field.fields)
#   v1 = return_value(b,f1)
#   allocate_parray(v1,length(field.op))
# end

# function Arrays.return_cache(
#   b::LagrangianDofBasis,
#   field::OperationField{<:PField})

#   f1 = OperationField(field.op[1],field.fields)
#   c1 = return_cache(b,f1)
#   a1 = evaluate!(c1,b,f1)
#   cache = Vector{typeof(c1)}(undef,length(field.op))
#   array = Vector{typeof(a1)}(undef,length(field.op))
#   for i = eachindex(cache)
#     fi = OperationField(field.op[i],field.fields)
#     cache[i] = return_cache(b,fi)
#   end
#   cache,PArray(array)
# end

# function Arrays.evaluate!(
#   cache,
#   b::LagrangianDofBasis,
#   field::OperationField{<:PField})

#   cf,array = cache
#   @inbounds for i = eachindex(array)
#     fi = OperationField(field.op[i],field.fields)
#     array[i] = evaluate!(cf[i],b,fi)
#   end
#   array
# end
abstract type PField <: Field end

Base.length(f::PField) = 1
Base.size(f::PField) = ()
Base.axes(f::PField) = ()
Base.IteratorSize(::Type{<:Field}) = Base.HasShape{0}()
Base.eltype(::Type{T}) where T<:Field = T
Base.iterate(af::PField) = (a,nothing)
Base.iterate(af::PField,::Nothing) = nothing
Base.getindex(af::PField,i::Integer) =  (@check i == 1; a)

struct PGenericField{T<:AbstractPFunction} <: PField
  object::T
end

CellData.GenericField(f::AbstractPFunction) = PField(f)

Base.length(f::PGenericField) = length(f.object)
Base.size(f::PGenericField) = (length(f),)
Base.iterate(f::PGenericField,it...) = iterate(f.object,it...)
Base.testitem(f::PGenericField) =

CellData.FieldGradient{N}(f::PField) where N = PField(FieldGradient{N}.(f.fields))
CellData.FieldGradient{N}(f::AbstractVector{<:PField}) where N = PField(FieldGradient{N}.(f.fields))
CellData.FieldGradient{N}(f::AbstractPFunction) where N = PField(get_fields(f,:FieldGradient;N))

CellData.ZeroField(f::AbstractPFunction) = PField(get_fields(f,:ZeroField))
