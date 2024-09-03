"""
    struct ReducedAlgebraicOperator{A} <: Projection end

Type representing a basis for a vector (sub)space (computed by compressing either
residual or jacobian snapshots) projected on a FESubspace. In practice:
1) we are given a manifold of parametrized residuals/jacobians, and a FESubspace
2) we compress the manifold, thus identifying a reduced subspace
3) we project the reduced subspace on the FESubspace by means of a (Petrov-)Galerkin
  projection

This Projection subtype is implemented in order to compute linear
combinations of ReducedAlgebraicOperator by vectors of coefficients just as in
the case of a standard matrix × vector product

"""
struct ReducedAlgebraicOperator{A} <: Projection
  basis::A
end

"""
    reduce_operator(red::AbstractReduction,b::Projection,r::FESubspace...) -> ReducedAlgebraicOperator

Computes a ReducedAlgebraicOperator from a Projection `b` and FESubspace(s) `r`.
When `b` consists of compressed residual snapshots, `r` stands for the `test` FE
space; when it consists of compressed jacobian snapshots, `r` stands for the
`trial` and `test` FE spaces

"""
function reduce_operator(red::AbstractReduction,b::Projection,r::FESubspace...)
  reduce_operator(b,map(get_basis,r)...)
end

function reduce_operator(b::PODBasis,b_test::PODBasis)
  bs = get_basis_space(b)
  bs_test = get_basis_space(b_test)
  b̂s = bs_test'*bs
  return ReducedAlgebraicOperator(b̂s)
end

function reduce_operator(b::PODBasis,b_trial::PODBasis,b_test::PODBasis)
  bs = get_basis_space(b)
  bs_trial = get_basis_space(b_trial)
  bs_test = get_basis_space(b_test)

  T = promote_type(eltype(bs_trial),eltype(bs_test))
  s = num_reduced_space_dofs(b_test),num_reduced_space_dofs(b),num_reduced_space_dofs(b_trial)
  b̂s = Array{T,3}(undef,s)

  @inbounds for i = 1:num_reduced_space_dofs(b)
    b̂s[:,i,:] = bs_test'*param_getindex(bs,i)*bs_trial
  end

  return ReducedAlgebraicOperator(b̂s)
end

# TT interface

function reduce_operator(b::TTSVDCores,b_test::TTSVDCores)
  b̂st = compress_cores(b,b_test)
  return ReducedAlgebraicOperator(b̂st)
end

function reduce_operator(b::TTSVDCores,b_trial::TTSVDCores,b_test::TTSVDCores)
  b̂st = compress_cores(b,b_trial,b_test)
  return ReducedAlgebraicOperator(b̂st)
end

function Base.:*(a::ReducedAlgebraicOperator{<:Matrix{T}},b::AbstractVector) where T
  return sum([a.basis[:,q]*b[q] for q = eachindex(b)])
end

function Base.:*(a::ReducedAlgebraicOperator{<:Array{T,3}},b::AbstractVector) where T
  return sum([a.basis[:,q,:]*b[q] for q = eachindex(b)])
end
