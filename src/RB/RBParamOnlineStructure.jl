struct RBParamOnlineStructure
  op::RBVariable
  assembler::Function
end

function RBParamOnlineStructure(op::RBVariable,assembler::Function)
  RBParamOnlineStructure(op,assembler)
end

function RBParamOnlineStructure(ad::RBAffineDecomposition;kwargs...)
  ad_eval = eval_affine_decomposition(ad)
  RBParamOnlineStructure(ad,ad_eval;kwargs...)
end

function RBParamOnlineStructure(
  ad::RBAffineDecomposition,
  ad_eval::AbstractArray;
  kwargs...)

  op = get_op(ad)
  basis = get_affine_decomposition(ad)
  coeff = compute_coefficient(op,basis;kwargs...)
  param_assembler(μ::Param) = rb_online_product(ad_eval,coeff(μ);nr=get_nrows(op))
  RBParamOnlineStructure(op,param_assembler)
end

function RBParamOnlineStructure(
  ad::RBAffineDecomposition{Nonlinear,Ttr,<:MDEIM},
  ad_eval::AbstractArray;
  kwargs...) where Ttr

  op = get_op(ad)
  basis = get_affine_decomposition(ad)
  coeff = compute_coefficient(op,basis;kwargs...)
  param_assembler(μ::Param,z) = rb_online_product(ad_eval,coeff(μ,z);nr=get_nrows(op))
  RBParamOnlineStructure(op,param_assembler)
end

function rb_online_product(
  basis::Matrix{Float},
  coeff::Vector{Float};
  nr=size(basis,1))

  @assert size(basis,2) == length(coeff) "Something is wrong"
  bc = sum([basis[:,k]*coeff[k] for k=eachindex(coeff)])
  reshape(bc,nr,:)
end

function rb_online_product(
  basis::Vector{Matrix{Float}},
  coeff::Vector{Matrix{Float}};
  nr=size(first(basis),1)*size(first(coeff),1))

  @assert length(basis) == length(coeff) "Something is wrong"
  bc = sum([kron(basis[k],coeff[k]) for k=eachindex(coeff)])
  reshape(bc,nr,:)
end

get_op(param_os::RBParamOnlineStructure) = param_os.op

get_id(param_os::RBParamOnlineStructure) = get_id(get_op(param_os))

function get_op(
  param_os::RBParamOnlineStructure,
  sym::Symbol)

  @assert get_id(param_os) == sym "Wrong operator"
  get_op(param_os)
end

function get_op(
  param_os::NTuple{N1,RBParamOnlineStructure},
  syms::NTuple{N2,Symbol}) where {N1,N2}

  all_param_os = Broadcasting(sym -> get_op(param_os,sym))(syms)
  filter(!isnothing,all_param_os)
end

get_online_structure(param_os::RBParamOnlineStructure) = param_os.assembler

function get_assembler(param_os::RBParamOnlineStructure,args...)
  get_online_structure(param_os)(args...)
end

function get_assembler(
  param_os::RBParamOnlineStructure{<:RBUnsteadyBilinVariable},
  args...)

  op = get_op(param_os)
  θ = get_θ(op)
  dt = get_dt(op)
  mat,mat_shift = get_online_structure(param_os)(args...)

  if get_id(op) == :M
    mat/dt - mat_shift/dt
  else
    θ*mat + (1-θ)*mat_shift
  end
end

function get_assembler(
  param_os::RBParamOnlineStructure{<:RBLiftVariable},
  args...)

  -get_online_structure(param_os)(args...)
end

function eval_assembler(
  param_os::NTuple{N1,RBParamOnlineStructure},
  syms::NTuple{N2,Symbol},
  args...) where {N1,N2}

  all_param_os = Broadcasting(sym -> eval_assembler(param_os,sym,args...))(syms)
  filter(!isnothing,all_param_os)
end
