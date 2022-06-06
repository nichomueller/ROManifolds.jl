
"""Computation of the inner product between 'vec1' and 'vec2', defined by the
  (positive definite) matrix 'norm_matrix'.
  If typeof(norm_matrix) == nothing (default), the standard inner product between
  'vec1' and 'vec2' is returned."""
function mydot(vec1::Vector, vec2::Vector, norm_matrix = nothing)

  if isnothing(norm_matrix)
      norm_matrix = float(I(size(vec1)[1]))
  end

  return sum(sum(vec1' * norm_matrix * vec2))

end

"""Computation of the norm of 'vec', defined by the (positive definite) matrix
'norm_matrix'. If typeof(norm_matrix) == nothing (default), the Euclidean norm
of 'vec' is returned."""
function mynorm(vec::Vector, norm_matrix = nothing)

  if isnothing(norm_matrix)
    norm_matrix = float(I(size(vec)[1]))
  end

  return sqrt(mydot(vec, vec, norm_matrix))

end

"""Computation of the standard matrix product between A, tensor of order 3,
  and B, tensor of order 2. If transpose_A is true, then the first two dimensions
  of A are permuted"""
function matrix_product!(AB::Array, A::Array, B::Array; transpose_A=false)

  @assert length(size(A)) == 3 && length(size(B)) == 2
  "Only implemented tensor order 3 * tensor order 2"

  if transpose_A
    return @tensor AB[i,j,k] = A[l,i,k] * B[l,j]
  else
    return @tensor AB[i,j,k] = A[i,l,k] * B[l,j]
  end

end

"""Generate a uniform random vector of dimension n between the ranges set by
  the vector of ranges 'a' and 'b'"""
function generate_Parameter(a::Vector, b::Vector, n::Int64 = 1)

  return [[rand(Uniform(a[i], b[i])) for i = 1:length(a)] for j in 1:n]

end
