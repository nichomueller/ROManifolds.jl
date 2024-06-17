abstract type ReducedAlgebraicOperator{A} <: Projection end

struct ReducedVectorOperator{A,B} <: ReducedAlgebraicOperator{A}
  mdeim_style::A
  basis::B
end

struct ReducedMatrixOperator{A,B} <: ReducedAlgebraicOperator{A}
  mdeim_style::A
  basis::B
end

function reduce_operator(mdeim_style::MDEIMStyle,b::Projection,r::RBSpace...;kwargs...)
  reduce_operator(mdeim_style,b,map(get_basis,r)...;kwargs...)
end

function reduce_operator(
  mdeim_style,
  b::PODBasis,
  b_test::PODBasis;
  kwargs...)

  bs = get_basis_space(b)
  bs_test = get_basis_space(b_test)
  b̂s = bs_test'*bs
  return ReducedVectorOperator(mdeim_style,b̂s)
end

function reduce_operator(
  mdeim_style,
  b::PODBasis,
  b_trial::PODBasis,
  b_test::PODBasis;
  kwargs...)

  bs = get_basis_space(b)
  bs_trial = get_basis_space(b_trial)
  bs_test = get_basis_space(b_test)

  T = promote_type(eltype(bs_trial),eltype(bs_test))
  s = num_reduced_space_dofs(b_test),num_reduced_space_dofs(b),num_reduced_space_dofs(b_trial)
  b̂s = Array{T,3}(undef,s)

  @inbounds for i = 1:num_reduced_space_dofs(b)
    b̂s[:,i,:] = bs_test'*param_getindex(bs,i)*bs_trial
  end

  return ReducedMatrixOperator(mdeim_style,b̂s)
end

# TT interface

function reduce_operator(
  mdeim_style,
  b::TTSVDCores,
  b_test::TTSVDCores;
  kwargs...)

  b̂st = compress_cores(b,b_test)
  return ReducedVectorOperator(mdeim_style,b̂st)
end

function reduce_operator(
  mdeim_style,
  b::TTSVDCores,
  b_trial::TTSVDCores,
  b_test::TTSVDCores;
  kwargs...)

  b̂st = compress_cores(b,b_trial,b_test;kwargs...)
  return ReducedMatrixOperator(mdeim_style,b̂st)
end

function compress_core(a::AbstractArray{T,3},btest::AbstractArray{S,3};kwargs...) where {T,S}
  TS = promote_type(T,S)
  ab = zeros(TS,size(btest,1),size(a,1),size(btest,3),size(a,3))
  @inbounds for i = CartesianIndices(size(ab))
    ib1,ia1,ib3,ia3 = Tuple(i)
    ab[i] = btest[ib1,:,ib3]'*a[ia1,:,ia3]
  end
  return ab
end

function compress_core(a::AbstractArray{T,4},btrial::AbstractArray{S,3},btest::AbstractArray{S,3};
  kwargs...) where {T,S}

  TS = promote_type(T,S)
  bab = zeros(TS,size(btest,1),size(a,1),size(btrial,1),size(btest,3),size(a,4),size(btrial,3))
  @inbounds for i = CartesianIndices(size(bab))
    ibV1,ia1,ibU1,ibV3,ia4,ibU3 = Tuple(i)
    bab[i] = btest[ibV1,:,ibV3]'*a[ia1,:,:,ia4]*btrial[ibU1,:,ibU3]
  end
  return bab
end

function multiply_cores(a::AbstractArray{T,4},b::AbstractArray{S,4}) where {T,S}
  @check (size(a,3)==size(b,1) && size(a,4)==size(b,2))
  TS = promote_type(T,S)
  ab = zeros(TS,size(a,1),size(a,2),size(b,3),size(b,4))
  @inbounds for i = CartesianIndices(size(ab))
    ia1,ia2,ib3,ib4 = Tuple(i)
    ab[i] = dot(a[ia1,ia2,:,:],b[:,:,ib3,ib4])
  end
  return ab
end

function multiply_cores(a::AbstractArray{T,6},b::AbstractArray{S,6}) where {T,S}
  @check (size(a,4)==size(b,1) && size(a,5)==size(b,2) && size(a,6)==size(b,3))
  TS = promote_type(T,S)
  ab = zeros(TS,size(a,1),size(a,2),size(a,3),size(b,4),size(b,5),size(b,6))
  @inbounds for i = CartesianIndices(size(ab))
    ia1,ia2,ia3,ib4,ib5,ib6 = Tuple(i)
    ab[i] = dot(a[ia1,ia2,ia3,:,:,:],b[:,:,:,ib4,ib5,ib6])
  end
  return ab
end

function multiply_cores(c1::AbstractArray,cores::AbstractArray...)
  _c1,_cores... = cores
  multiply_cores(multiply_cores(c1,_c1),_cores...)
end

function _dropdims(a::AbstractArray{T,4}) where T
  @check size(a,1) == size(a,2) == 1
  dropdims(a;dims=(1,2))
end

function _dropdims(a::AbstractArray{T,6}) where T
  @check size(a,1) == size(a,2) == size(a,3) == 1
  dropdims(a;dims=(1,2,3))
end

function compress_cores(core::TTSVDCores,bases::TTSVDCores...;kwargs...)
  ccores = map((a,b...)->compress_core(a,b...;kwargs...),get_cores(core),get_cores.(bases)...)
  ccore = multiply_cores(ccores...)
  _dropdims(ccore)
end

function Base.:*(a::ReducedVectorOperator,b::AbstractVector)
  return sum([a.basis[:,q]*b[q] for q = eachindex(b)])
end

function Base.:*(a::ReducedMatrixOperator,b::AbstractVector)
  return sum([a.basis[:,q,:]*b[q] for q = eachindex(b)])
end
