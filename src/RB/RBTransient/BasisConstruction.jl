# tt-svd

function RBSteady.projection(
  red::TTSVDReduction,
  A::AbstractTransientSnapshots{T,N},
  X::AbstractRankTensor) where {T,N}

  red_style = ReductionStyle(red)
  cores,remainder = ttsvd(red_style,A,X)
  core_t,remainder_t = RBSteady.ttsvd_loop(red_style[N-1],remainder)
  push!(cores,core_t)

  return cores
end

function check_orthogonality(cores::AbstractVector{<:AbstractArray{T,3}},X::AbstractTProductTensor) where T
  Xglobal_space = kron(X)
  cores_space...,core_time = cores
  basis_space = cores2basis(cores_space...)
  isorth_space = norm(basis_space'*Xglobal_space*basis_space - I) ≤ 1e-10
  num_times = size(core_time,2)
  Xglobal_spacetime = kron(Float64.(I(num_times)),Xglobal_space)
  basis_spacetime = cores2basis(cores...)
  isorth_spacetime = norm(basis_spacetime'*Xglobal_spacetime*basis_spacetime - I) ≤ 1e-10
  return isorth_space,isorth_spacetime
end
