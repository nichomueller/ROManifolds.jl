struct RBParamOnlineStructure
  op::RBVariable
  assembler::Function
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
  assembler(μ::Param) = rb_online_product(ad_eval,coeff(μ);nr=get_nrows(op))
  RBParamOnlineStructure(op,assembler)
end

function RBParamOnlineStructure(
  ad::RBAffineDecomposition{Nonlinear,Ttr,<:MDEIM},
  ad_eval::AbstractArray;
  kwargs...) where Ttr

  op = get_op(ad)
  basis = get_affine_decomposition(ad)
  coeff = compute_coefficient(op,basis;kwargs...)
  assembler(μ::Param,z) = rb_online_product(ad_eval,coeff(μ,z);nr=get_nrows(op))
  RBParamOnlineStructure(op,assembler)
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

get_assembler(param_os::RBParamOnlineStructure) = param_os.assembler

function assemble(
  ::RBVariable,
  assembler::Function,
  args...)

  assembler(args...)
end

function assemble(
  ::RBLiftVariable,
  assembler::Function,
  args...)

  -assembler(args...)
end

function assemble(param_os::RBParamOnlineStructure,args...)::Matrix{Float}
  op = get_op(param_os)
  assembler = get_assembler(param_os)
  assemble(op,assembler,args...)
end

function assemble(
  param_os::NTuple{N1,RBParamOnlineStructure},
  sym::Symbol,
  args...) where N1

  syms = get_id.(get_op.(param_os))
  idx = findall(x -> x == sym,syms)[1]
  op = get_op(param_os[idx])
  assembler = get_assembler(param_os[idx])
  assemble(op,assembler,args...)
end

function assemble(
  param_os::NTuple{N1,RBParamOnlineStructure},
  syms::NTuple{N2,Symbol},
  args...) where {N1,N2}

  assemble(sym::Symbol) = assemble(param_os,sym,args...)
  assemble.(syms)
end
