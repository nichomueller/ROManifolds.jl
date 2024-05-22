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
  mdeim_style::SpaceOnlyMDEIM,
  b::PODBasis,
  b_test::PODBasis;
  kwargs...)

  bs = get_basis_space(b)
  bs_test = get_basis_space(b_test)
  bt_test = get_basis_time(b_test)

  V = Vector{eltype(bs)}
  b̂st = Vector{Vector{V}}(undef,num_reduced_space_dofs(b))

  b̂s = bs_test'*bs
  @inbounds for i = eachindex(b̂st)
    b̂si = b̂s[:,i]
    b̂st[i] = eachcol(kronecker(bt_test',b̂si))
  end

  return ReducedVectorOperator(mdeim_style,b̂st)
end

function reduce_operator(
  mdeim_style::SpaceOnlyMDEIM,
  b::PODBasis,
  b_trial::PODBasis,
  b_test::PODBasis;
  kwargs...)

  bs = get_basis_space(b)
  bs_trial = get_basis_space(b_trial)
  bt_trial = get_basis_time(b_trial)
  bs_test = get_basis_space(b_test)
  bt_test = get_basis_time(b_test)

  M = Matrix{eltype(bs)}
  b̂st = Vector{Vector{M}}(undef,num_reduced_space_dofs(b))

  b̂t = combine_basis_time(bt_trial,bt_test;kwargs...)

  @inbounds for i = eachindex(b̂st)
    b̂si = bs_test'*get_values(bs)[i]*bs_trial
    b̂st[i] = map(k->kronecker(b̂t[k,:,:],b̂si),axes(b̂t,1))
  end

  return ReducedMatrixOperator(mdeim_style,b̂st)
end

function reduce_operator(
  mdeim_style::SpaceTimeMDEIM,
  b::PODBasis,
  b_test::PODBasis;
  kwargs...)

  bs = get_basis_space(b)
  bt = get_basis_time(b)
  bs_test = get_basis_space(b_test)
  bt_test = get_basis_time(b_test)

  V = Vector{eltype(bs)}
  b̂st = Vector{V}(undef,num_reduced_dofs(b))

  b̂s = bs_test'*bs
  b̂t = bt_test'*bt
  b̂st .= eachcol(kronecker(b̂t,b̂s))

  return ReducedVectorOperator(mdeim_style,b̂st)
end

function reduce_operator(
  mdeim_style::SpaceTimeMDEIM,
  b::PODBasis,
  b_trial::PODBasis,
  b_test::PODBasis;
  kwargs...)

  bs = get_basis_space(b)
  bt = get_basis_time(b)
  bs_trial = get_basis_space(b_trial)
  bt_trial = get_basis_time(b_trial)
  bs_test = get_basis_space(b_test)
  bt_test = get_basis_time(b_test)

  M = Matrix{eltype(bs)}
  b̂st = Vector{M}(undef,num_reduced_dofs(b))

  b̂t = combine_basis_time(bt,bt_trial,bt_test;kwargs...)

  @inbounds for is = 1:num_reduced_space_dofs(b)
    b̂si = bs_test'*get_values(bs)[is]*bs_trial
    for it = 1:num_reduced_times(b)
      ist = (it-1)*num_reduced_space_dofs(b)+is
      b̂ti = b̂t[it]
      b̂st[ist] = kronecker(b̂ti,b̂si)
    end
  end

  return ReducedMatrixOperator(mdeim_style,b̂st)
end

# TT interface

function reduce_operator(
  mdeim_style::SpaceTimeMDEIM,
  b::TTSVDCores,
  b_test::TTSVDCores;
  kwargs...)

  b̂st = compress_cores(b,b_test)
  return ReducedVectorOperator(mdeim_style,b̂st)
end

function reduce_operator(
  mdeim_style::SpaceTimeMDEIM,
  b::TTSVDCores,
  b_trial::TTSVDCores,
  b_test::TTSVDCores;
  kwargs...)

  b̂st = compress_cores(b,b_trial,b_test;kwargs...)
  return ReducedMatrixOperator(mdeim_style,b̂st)
end

function compress_core(a::Array{T,3},btest::Array{S,3};kwargs...) where {T,S}
  TS = promote_type(T,S)
  ab = zeros(TS,size(btest,1),size(a,1),size(btest,3),size(a,3))
  @inbounds for i = CartesianIndices(size(ab))
    ib1,ia1,ib3,ia3 = Tuple(i)
    ab[i] = btest[ib1,:,ib3]'*a[ia1,:,ia3]
  end
  return ab
end

function compress_core(a::Array{T,4},btrial::Array{S,3},btest::Array{S,3};kwargs...) where {T,S}
  TS = promote_type(T,S)
  bab = zeros(TS,size(btest,1),size(a,1),size(btrial,1),size(btest,3),size(a,4),size(btrial,3))
  @inbounds for i = CartesianIndices(size(bab))
    ibV1,ia1,ibU1,ibV3,ia4,ibU3 = Tuple(i)
    bab[i] = btest[ibV1,:,ibV3]'*a[ia1,:,:,ia4]*btrial[ibU1,:,ibU3]
  end
  return bab
end

function compress_core(a::Array{T,3},btrial::Array{S,3},btest::Array{S,3};combine=(x,y)->x) where {T,S}
  TS = promote_type(T,S)
  bab = zeros(TS,size(btest,1),size(a,1),size(btrial,1),size(btest,3),size(a,4),size(btrial,3))
  bab_shift = zeros(TS,size(btest,1),size(a,1),size(btrial,1),size(btest,3),size(a,4),size(btrial,3))
  @inbounds for i = CartesianIndices(size(bab))
    ibV1,ia1,ibU1,ibV3,ia4,ibU3 = Tuple(i)
    bab[i] = sum(btest[ibV1,:,ibV3].*a[ia1,:,ia4].*btrial[ibU1,:,ibU3])
    bab_shift[i] = sum(btest[ibV1,2:end,ibV3].*a[ia1,2:end,ia4].*btrial[ibU1,1:end-1,ibU3])
  end
  return combine(bab,bab_shift)
end

function multiply_cores(a::Array{T,4},b::Array{S,4}) where {T,S}
  @check (size(a,3)==size(b,1) && size(a,4)==size(b,2))
  TS = promote_type(T,S)
  ab = zeros(TS,size(a,1),size(a,2),size(b,3),size(b,4))
  @inbounds for i = CartesianIndices(size(ab))
    ia1,ia2,ib3,ib4 = Tuple(i)
    ab[i] = dot(a[ia1,ia2,:,:],b[:,:,ib3,ib4])
  end
  return ab
end

function multiply_cores(a::Array{T,6},b::Array{S,6}) where {T,S}
  @check (size(a,4)==size(b,1) && size(a,5)==size(b,2) && size(a,6)==size(b,3))
  TS = promote_type(T,S)
  ab = zeros(TS,size(a,1),size(a,2),size(a,3),size(b,4),size(b,5),size(b,6))
  @inbounds for i = CartesianIndices(size(ab))
    ia1,ia2,ia3,ib4,ib5,ib6 = Tuple(i)
    ab[i] = dot(a[ia1,ia2,ia3,:,:,:],b[:,:,:,ib4,ib5,ib6])
  end
  return ab
end

function multiply_cores(c1::Array,cores::Array...)
  _c1,_cores... = cores
  multiply_cores(multiply_cores(c1,_c1),_cores...)
end

function _dropdims(a::Array{T,4}) where T
  @check size(a,1) == size(a,2) == 1
  dropdims(a;dims=(1,2))
end

function _dropdims(a::Array{T,6}) where T
  @check size(a,1) == size(a,2) == size(a,3) == 1
  dropdims(a;dims=(1,2,3))
end

function compress_cores(core::TTSVDCores,bases::TTSVDCores...;kwargs...)
  ccores = map((a,b...)->compress_core(a,b...;kwargs...),get_cores(core),get_cores.(bases)...)
  ccore = multiply_cores(ccores...)
  _dropdims(ccore)
end

function combine_basis_time(B::AbstractMatrix,C::AbstractMatrix;combine=(x,y)->x)
  time_ndofs = size(C,1)
  nt_row = size(C,2)
  nt_col = size(B,2)

  T = eltype(B)
  bt_proj = zeros(T,time_ndofs,nt_row,nt_col)
  bt_proj_shift = copy(bt_proj)
  @inbounds for jt = 1:nt_col, it = 1:nt_row
    bt_proj[:,it,jt] .= C[:,it].*B[:,jt]
    bt_proj_shift[2:end,it,jt] .= C[2:end,it].*B[1:end-1,jt]
  end

  combine(bt_proj,bt_proj_shift)
end

function combine_basis_time(A::AbstractMatrix,B::AbstractMatrix,C::AbstractMatrix;kwargs...)
  map(a->combine_basis_time(a,B,C;kwargs...),eachcol(A))
end

function combine_basis_time(a::AbstractVector,B::AbstractMatrix,C::AbstractMatrix;combine=(x,y)->x)
  nt_row = size(C,2)
  nt_col = size(B,2)

  T = eltype(B)
  bt_proj = zeros(T,nt_row,nt_col)
  bt_proj_shift = copy(bt_proj)
  @inbounds for jt = 1:nt_col, it = 1:nt_row
    bt_proj[it,jt] = sum(C[:,it].*a.*B[:,jt])
    bt_proj_shift[it,jt] = sum(C[2:end,it].*a[2:end].*B[1:end-1,jt])
  end

  combine(bt_proj,bt_proj_shift)
end

function Base.:*(a::ReducedVectorOperator{SpaceOnlyMDEIM},b::AbstractMatrix)
  return sum([a.basis[q]*b[q,:]' for q = eachindex(a.basis)])
end

function Base.:*(a::ReducedMatrixOperator{SpaceOnlyMDEIM},b::AbstractMatrix)
  return sum([a.basis[q][k]*b[q,k] for q = eachindex(a.basis) for k = eachindex(a.basis[q])])
end

function Base.:*(a::ReducedAlgebraicOperator{SpaceTimeMDEIM},b::AbstractVector)
  return sum([a.basis[q]*b[q] for q = eachindex(a.basis)])
end

#TODO try to unify the two multiplications
# this is the case where a TT strategy is employed

function Base.:*(a::ReducedMatrixOperator{SpaceTimeMDEIM,Array{T,3}},b::AbstractVector) where T
  return sum([a.basis[:,q,:]*b[q] for q = eachindex(b)])
end

function Base.:*(a::ReducedVectorOperator{SpaceTimeMDEIM,Matrix{T}},b::AbstractVector) where T
  return sum([a.basis[:,q]*b[q] for q = eachindex(b)])
end
