mutable struct RBParamOnlineStructure{Top}
  op::Top
  on_structure::Function
end

function RBParamOnlineStructure(
  op::Top,
  on_structure::Function) where Top

  RBParamOnlineStructure{Top}(op,on_structure)
end

function RBParamOnlineStructure(
  rb_structure::Tuple,
  os::Tuple;
  kwargs...)

  Broadcasting((rb,off)->
    RBParamOnlineStructure(rb,off;kwargs...))(rb_structure,os)
end

function RBParamOnlineStructure(
  rb_structure::Union{RBAffineStructure,RBNonaffineStructure},
  os::AbstractArray;
  kwargs...)

  op = get_op(rb_structure)
  basis = get_offline_structure(rb_structure)
  coeff = compute_coefficient(op,basis;kwargs...)
  param_on_structure(μ::Param) = rb_online_product(os,coeff(μ);nr=get_nrows(op))

  RBParamOnlineStructure(op,param_on_structure)
end

function RBParamOnlineStructure(
  rb_structure::RBNonlinearStructure,
  os::Union{AbstractArray,NTuple{2,AbstractArray}};
  kwargs...)

  op = get_op(rb_structure)
  basis = get_offline_structure(rb_structure)
  coeff = compute_coefficient(op,basis;kwargs...)
  param_on_structure(u::Vector{Float}) = rb_online_product(os,coeff(u);nr=get_nrows(op))

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
  coeff::Vector{Float};
  nr=size(basis,1))

  @assert length(basis) == length(coeff) "Something is wrong"
  bc = sum([basis[k]*coeff[k] for k=eachindex(coeff)])
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

function rb_online_product(basis,coeff::NTuple{2,T};kwargs...) where T
  Broadcasting(c->rb_online_product(basis,c;kwargs...))(coeff)
end

function rb_online_product(basis::NTuple{2,T},coeff;kwargs...) where T
  Broadcasting(b->rb_online_product(b,coeff;kwargs...))(basis)
end

get_op(param_os::RBParamOnlineStructure) = param_os.op
get_id(param_os::RBParamOnlineStructure) = get_id(get_op(param_os))

function get_op(
  param_os::RBParamOnlineStructure,
  sym::Symbol)

  get_id(param_os) == sym ? param_os.op : return
end

function get_op(
  param_os::NTuple{N,RBParamOnlineStructure},
  sym::Symbol) where N

  ids = get_id.(param_os)
  idx = findall(x -> x == sym,ids)
  !isempty(idx) ? get_op(param_os[first(idx)],sym) : return
end

function get_op(
  param_os::NTuple{N1,RBParamOnlineStructure},
  syms::NTuple{N2,Symbol}) where {N1,N2}

  all_param_os = Broadcasting(sym -> get_op(param_os,sym))(syms)
  filter(!isnothing,all_param_os)
end

get_online_structure(param_os::RBParamOnlineStructure) = param_os.on_structure

function eval_on_structure(param_os::RBParamOnlineStructure,q)
  get_online_structure(param_os)(q)
end

function eval_on_structure(
  param_os::RBParamOnlineStructure{<:RBUnsteadyBilinVariable},
  q)

  op = get_op(param_os)
  θ = get_θ(op)
  dt = get_dt(op)
  mat,mat_shift = get_online_structure(param_os)(q)

  if get_id(op) == :M
    mat/dt - mat_shift/dt
  else
    θ*mat + (1-θ)*mat_shift
  end
end

function eval_on_structure(
  param_os::RBParamOnlineStructure{<:RBLiftVariable},
  q)

  -get_online_structure(param_os)(q)
end

function eval_on_structure(
  param_os::RBParamOnlineStructure,
  sym::Symbol,
  q)

  get_id(param_os) == sym ? eval_on_structure(param_os,q) : return
end

function eval_on_structure(
  param_os::NTuple{N,RBParamOnlineStructure},
  sym::Symbol,
  q) where N

  ids = get_id.(param_os)
  idx = findall(x -> x == sym,ids)
  !isempty(idx) ? eval_on_structure(param_os[first(idx)],sym,q) : return
end

function eval_on_structure(
  param_os::NTuple{N1,RBParamOnlineStructure},
  syms::NTuple{N2,Symbol},
  q) where {N1,N2}

  all_param_os = Broadcasting(sym -> eval_on_structure(param_os,sym,q))(syms)
  filter(!isnothing,all_param_os)
end
