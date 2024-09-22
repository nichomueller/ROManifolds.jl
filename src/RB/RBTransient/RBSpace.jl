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

function RBSteady.project(r::TransientEvalFESubspace,x::AbstractParamVector)
  x̂ = allocate_in_domain(r)
  np = num_params(r.realization)
  nt = num_times(r.realization)
  rsub = RBSteady.get_reduced_subspace(r)
  @inbounds for ip in eachindex(x)
    ipt = ip:nt:np*nt
    x̂[ip] = project(rsub,x[ipt])
  end
  return x̂
end

function RBSteady.inv_project(r::TransientEvalFESubspace,x̂::AbstractParamVector)
  x = allocate_in_range(r)
  np = num_params(r.realization)
  nt = num_times(r.realization)
  rsub = RBSteady.get_reduced_subspace(r)
  @inbounds for ip in eachindex(x̂)
    Xip = inv_project(rsub,x̂[ip])
    for it in 1:nt
      x[ip+(it-1)*np] = Xip[:,it]
    end
  end
  return x
end
