"""
    abstract type ReducedAlgebraicOperator{A} <: Projection end

Type representing a basis for a vector (sub)space (computed by compressing either
residual or jacobian snapshots) projected on a FESubspace. In practice:
1) we are given a manifold of parametrized residuals/jacobians, and a FESubspace
2) we compress the manifold, thus identifying a reduced subspace
3) we project the reduced subspace on the FESubspace by means of a (Petrov-)Galerkin
  projection

This Projection subtype is implemented for two reasons:

- a field containing the mdeim style is provided for multiple dispatching
- a specialization for Base.* is implemented in order to compute linear
  combinations of ReducedAlgebraicOperator by vectors of coefficients just as in
  the case of a standard matrix × vector product

Subtypes:
- [`ReducedVectorOperator`](@ref)
- [`ReducedMatrixOperator`](@ref)

"""
abstract type ReducedAlgebraicOperator <: Projection end

"""
"""
struct ReducedVectorOperator{A} <: ReducedAlgebraicOperator
  basis::A
end

"""
"""
struct ReducedMatrixOperator{A} <: ReducedAlgebraicOperator
  basis::B
end

"""
    reduce_operator(b::Projection,r::FESubspace...;kwargs...) -> ReducedAlgebraicOperator

Computes a ReducedAlgebraicOperator from a Projection `b` and FESubspace(s) `r`.
When `b` consists of compressed residual snapshots, `r` stands for the `test` FE
space; when it consists of compressed jacobian snapshots, `r` stands for the
`trial` and `test` FE spaces

"""
function reduce_operator(b::Projection,r::FESubspace...;kwargs...)
  reduce_operator(b,map(get_basis,r)...;kwargs...)
end

function reduce_operator(
  b::PODBasis,
  b_test::PODBasis;
  kwargs...)

  bs = get_basis_space(b)
  bs_test = get_basis_space(b_test)
  b̂s = bs_test'*bs
  return ReducedVectorOperator(b̂s)
end

function reduce_operator(b::PODBasis,b_trial::PODBasis,b_test::PODBasis;kwargs...)
  bs = get_basis_space(b)
  bs_trial = get_basis_space(b_trial)
  bs_test = get_basis_space(b_test)

  T = promote_type(eltype(bs_trial),eltype(bs_test))
  s = num_reduced_space_dofs(b_test),num_reduced_space_dofs(b),num_reduced_space_dofs(b_trial)
  b̂s = Array{T,3}(undef,s)

  @inbounds for i = 1:num_reduced_space_dofs(b)
    b̂s[:,i,:] = bs_test'*param_getindex(bs,i)*bs_trial
  end

  return ReducedMatrixOperator(b̂s)
end

# TT interface

function reduce_operator(b::TTSVDCores,b_test::TTSVDCores;kwargs...)
  b̂st = compress_cores(b,b_test)
  return ReducedVectorOperator(b̂st)
end

function reduce_operator(b::TTSVDCores,b_trial::TTSVDCores,b_test::TTSVDCores;kwargs...)
  b̂st = compress_cores(b,b_trial,b_test;kwargs...)
  return ReducedMatrixOperator(b̂st)
end

function Base.:*(a::ReducedVectorOperator,b::AbstractVector)
  return sum([a.basis[:,q]*b[q] for q = eachindex(b)])
end

function Base.:*(a::ReducedMatrixOperator,b::AbstractVector)
  return sum([a.basis[:,q,:]*b[q] for q = eachindex(b)])
end
