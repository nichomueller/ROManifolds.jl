mutable struct RBParamOnlineStructure{Top}
  op::Top
  on_structure::Function
end

function RBParamOnlineStructure(
  rb_structure::Tuple,
  args...)

  ntup_rb_structure = expand(rb_structure)
  RBParamOnlineStructure(ntup_rb_structure,args...)
end

function RBParamOnlineStructure(
  rb_structure::NTuple{N,RBOfflineStructure},
  args...) where N

  Broadcasting(rb->RBParamOnlineStructure(rb,args...))(rb_structure)
end

function RBParamOnlineStructure(
  rb_structure::RBOfflineStructure,
  args...;
  kwargs...)

  op = get_op(rb_structure)
  basis = get_offline_structure(rb_structure)
  coeff = compute_coefficient(op,basis;kwargs...)
  os = eval_off_structure(rb_structure,args...)
  param_on_structure(μ) = rb_online_product(os,coeff(μ);nr=get_nrows(op))

  RBParamOnlineStructure(op,param_on_structure)
end

function RBParamOnlineStructure(
  rb_structure::RBOfflineStructure{RBVariable{Nonlinear,Ttr}},
  args...;
  kwargs...) where Ttr

  op = get_op(rb_structure)
  basis = get_offline_structure(rb_structure)
  coeff = compute_coefficient(op,basis;kwargs...)
  os = eval_off_structure(rb_structure,args...)
  param_on_structure(u1,u2) = rb_online_product(os,coeff(u1,u2);nr=get_nrows(op))

  RBParamOnlineStructure(op,param_on_structure)
end

function rb_online_product(
  basis::Matrix{Float},
  coeff::Vector{Float};
  nr=size(basis,1))

  @assert size(basis,2) == length(coeff) "Something is wrong"
  bc = sum([basis[:,k]*coeff[k] for k=eachindex(coeff)])
  Matrix(reshape(bc,nr,:))
end

function rb_online_product(
  basis::BlockMatrix{Float},
  coeff::BlockMatrix{Float};
  nr=size(first(basis),1)*size(first(coeff),1))

  @assert length(basis) == length(coeff) "Something is wrong"
  bc = sum([kron(basis[k],coeff[k]) for k=eachindex(coeff)])
  Matrix(reshape(bc,nr,:))
end

function rb_online_product(
  basis::BlockMatrix{Float},
  coeff::Coefficient;
  kwargs...)

  rb_online_product(basis,get_coefficient(coeff);kwargs...)
end

function rb_online_product(basis,coeff::NTuple{2,T};kwargs...) where T
  Broadcasting(c->rb_online_product(basis,c;kwargs...))(coeff)
end
