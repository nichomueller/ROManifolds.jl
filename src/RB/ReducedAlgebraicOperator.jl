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
  b::Projection,
  b_test::Projection;
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
  b::Projection,
  b_trial::Projection,
  b_test::Projection;
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
  b::Projection,
  b_test::Projection;
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
  b::Projection,
  b_trial::Projection,
  b_test::Projection;
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
