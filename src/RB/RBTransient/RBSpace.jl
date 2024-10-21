function ODEs.time_derivative(r::FESubspace)
  fet = time_derivative(get_fe_space(r))
  rb = get_reduced_subspace(r)
  fe_subspace(fet,rb)
end

function RBSteady.project(r1::FESubspace,x::Projection,r2::FESubspace,combine::Function)
  galerkin_projection(RBSteady.get_reduced_subspace(r1),x,RBSteady.get_reduced_subspace(r2),combine)
end

const TransientEvalRBSpace{A<:FESubspace} = EvalRBSpace{A,<:TransientRealization}

_change_length(::Type{T},r::TransientRealization) where T = T

function _change_length(
  ::Type{ConsecutiveParamVector{T,L}},
  r::TransientRealization
  ) where {T,L}

  ConsecutiveParamVector{T,Int(L/num_times(r))}
end

function _change_length(
  ::Type{<:BlockParamVector{T,L}},
  r::TransientRealization
  ) where {T,L}

  BlockParamVector{T,Int(L/num_times(r))}
end

function FESpaces.get_vector_type(r::TransientEvalRBSpace)
  V = get_vector_type(r.subspace)
  return _change_length(V,r.realization)
end

function RBSteady.project!(x̂,r::TransientEvalRBSpace,x::ConsecutiveParamVector)
  np = num_params(r.realization)
  nt = num_times(r.realization)
  rsub = RBSteady.get_reduced_subspace(r)
  @inbounds for ip in eachindex(x̂)
    ipt = ip:np:np*nt
    xpt = vec(x.data[:,ipt])
    x̂[ip] = project(rsub,xpt)
  end
  return x̂
end

function RBSteady.inv_project!(x,r::TransientEvalRBSpace,x̂::AbstractParamVector)
  np = num_params(r.realization)
  nt = num_times(r.realization)
  rsub = RBSteady.get_reduced_subspace(r)
  @inbounds for ip in eachindex(x̂)
    Xip = inv_project(rsub,x̂[ip])
    if ndims(Xip) == 1
      Xip = reshape(Xip,:,nt)
    end
    for it in 1:nt
      x[ip+(it-1)*np] = Xip[:,it]
    end
  end
  return x
end

const TransientEvalMultiFieldRBSpace = EvalMultiFieldRBSpace{<:TransientRealization}

for S in (:BlockVector,:BlockParamVector), f! in (:(RBSteady.project!),:(RBSteady.inv_project!))
  @eval begin
    function $f!(y::$S,r::TransientEvalMultiFieldRBSpace,x::$S)
      for i in 1:blocklength(x)
        $f!(y[Block(i)],r[i],x[Block(i)])
      end
      return y
    end
  end
end
