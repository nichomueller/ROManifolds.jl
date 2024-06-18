function RBSteady.reduced_basis(feop::TransientParamFEOperator,s,norm_matrix;kwargs...)
  reduced_basis(s,norm_matrix;kwargs...)
end

function RBSteady.reduced_basis(feop::TransientParamSaddlePointFEOp,s,norm_matrix;kwargs...)
  bases = reduced_basis(feop.op,s,norm_matrix;kwargs...)
  enrich_basis(feop,bases,norm_matrix)
end

function RBSteady.reduced_basis(feop::TransientParamFEOperatorWithTrian,s,norm_matrix;kwargs...)
  reduced_basis(feop.op,s,norm_matrix;kwargs...)
end

function RBSteady.reduced_basis(feop::TransientParamLinearNonlinearFEOperator,s,norm_matrix;kwargs...)
  reduced_basis(join_operators(feop),s,norm_matrix;kwargs...)
end

const TransientRBSpace{A,B<:TransientProjection} = RBSpace{A,B}

(U::TransientRBSpace)(μ,t) = evaluate(U,μ,t)

ODEs.time_derivative(U::TransientRBSpace) = RBSpace(time_derivative(U),U.basis)

get_basis_time(r::TransientRBSpace) = get_basis_time(r.basis)
ParamDataStructures.num_times(r::TransientRBSpace) = num_times(r.basis)
num_reduced_times(r::TransientRBSpace) = num_reduced_times(r.basis)

function FESpaces.get_vector_type(r::TransientRBSpace)
  change_length(x) = x
  change_length(::Type{VectorOfVectors{T,L}}) where {T,L} = VectorOfVectors{T,Int(L/num_times(r))}
  change_length(::Type{<:BlockVectorOfVectors{T,L}}) where {T,L} = BlockVectorOfVectors{T,Int(L/num_times(r))}
  V = get_vector_type(r.space)
  newV = change_length(V)
  return newV
end

function Arrays.evaluate!(cache,k::RBSteady.RecastMap,x::AbstractParamVector,r::TransientRBSpace)
  @inbounds for ip in eachindex(x)
    Xip = recast(x[ip],r.basis)
    for it in 1:num_times(r)
      cache[(it-1)*length(x)+ip] .= Xip[:,it]
    end
  end
end

function RBSteady.pod_error(r::RBSpace,s::AbstractTransientSnapshots,norm_matrix::AbstractMatrix)
  s2 = change_mode(s)
  basis_space = get_basis_space(r)
  basis_time = get_basis_time(r)
  err_space = norm(s - basis_space*basis_space'*norm_matrix*s) / norm(s)
  err_time = norm(s2 - basis_time*basis_time'*s2) / norm(s2)
  Dict("err_space"=>err_space,"err_time"=>err_time)
end
