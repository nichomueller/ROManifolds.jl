function ODEs.time_derivative(r::FESubspace)
  fet = time_derivative(get_fe_space(r))
  rb = get_reduced_subspace(r)
  fe_subspace(fet,rb)
end

function RBSteady.project(r1::FESubspace,x::Projection,r2::FESubspace,combine::Function)
  galerkin_projection(RBSteady.get_reduced_subspace(r1),x,RBSteady.get_reduced_subspace(r2),combine)
end

const TransientEvalFESubspace{A<:FESubspace} = EvalFESubspace{A,<:TransientRealization}

_change_length(::Type{T},r::TransientRealization) where T = T

function _change_length(
  ::Type{<:ConsecutiveVectorOfVectors{T,L}},
  r::TransientRealization
  ) where {T,L}

  ConsecutiveVectorOfVectors{T,Int(L/num_times(r))}
end

function _change_length(
  ::Type{<:BlockVectorOfVectors{T,L}},
  r::TransientRealization
  ) where {T,L}

  BlockVectorOfVectors{T,Int(L/num_times(r))}
end

function FESpaces.get_vector_type(r::TransientEvalFESubspace)
  V = get_vector_type(r.subspace)
  return _change_length(V,r.realization)
end
