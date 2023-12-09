function expand(tup::Tuple)
  t = ()
  for el = tup
    if isa(el,Tuple)
      t = (t...,expand(el)...)
    else
      t = (t...,el)
    end
  end
  t
end

function compress_array(entire_array::AbstractMatrix)
  sum_cols = vec(sum(entire_array,dims=2))
  nonzero_idx = findall(x -> abs(x) ≥ eps(),sum_cols)
  nonzero_idx,entire_array[nonzero_idx,:]
end

function compress_array(entire_array::SparseMatrixCSC)
  findnz(entire_array[:])
end

function recenter(vec::Vector{<:AbstractVector},vec0::Vector;θ=0.5)
  vecθ = θ*vec + (1-θ)*[vec0,vec[1:end-1]...]
  return vecθ
end

LinearAlgebra.norm(v::AbstractVector,::Nothing) = norm(v)

LinearAlgebra.norm(v::AbstractVector,X::AbstractMatrix) = sqrt(v'*X*v)
