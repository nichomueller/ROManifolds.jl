function ODEs.time_derivative(r::FESubspace)
  fet = time_derivative(get_space(r))
  rb = get_reduced_subspace(r)
  fe_subspace(fet,rb)
end

const TransientEvalFESubspace{A<:FESubspace} = EvalFESubspace{A,TransientRealization}

_change_length(::Type{T},r::TransientRealization) where T = T

function _change_length(
  ::Type{ConsecutiveVectorOfVectors{T,L}},
  r::TransientRealization
  ) where {T,L}

  ConsecutiveVectorOfVectors{T,Int(L/num_times(r))}
end

function _change_length(
  ::Type{BlockVectorOfVectors{T,L}},
  r::TransientRealization
  ) where {T,L}

  BlockVectorOfVectors{T,Int(L/num_times(r))}
end

function FESpaces.get_vector_type(r::AbstractTransientRBSpace)
  V = get_vector_type(r.subspace)
  return _change_length(V)
end
