struct RBParamOnlineStructure{Top}
  op::Top
  on_structure::Function
end

function RBParamOnlineStructure(
  op::Top,
  on_structure::Function) where Top

  RBParamOnlineStructure{Top}(op,on_structure)
end

function RBParamOnlineStructure(
  ad::Tuple,
  ad_eval::Tuple;
  kwargs...)

  ad_ntupl = expand(ad)
  ad_eval_ntupl = expand(ad_eval)

  Broadcasting((ad_i,ad_eval_i)->
    RBParamOnlineStructure(ad_i,ad_eval_i;kwargs...))(ad_ntupl,ad_eval_ntupl)
end

function RBParamOnlineStructure(
  ad::RBAffineDecomposition,
  ad_eval::AbstractArray;
  kwargs...)

  op = get_op(ad)
  basis = get_affine_decomposition(ad)
  coeff = compute_coefficient(op,basis;kwargs...)
  param_on_structure(μ::Param) =
    rb_online_product(ad_eval,coeff(μ);nr=get_nrows(op))

  RBParamOnlineStructure(op,param_on_structure)
end

function RBParamOnlineStructure(
  ad::RBAffineDecomposition{Nonlinear,Ttr,<:MDEIM},
  ad_eval::AbstractArray;
  kwargs...) where Ttr

  op = get_op(ad)
  basis = get_affine_decomposition(ad)
  coeff = compute_coefficient(op,basis;kwargs...)
  param_on_structure(μ::Param,z) =
    rb_online_product(ad_eval,coeff(μ,z);nr=get_nrows(op))

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

function get_on_structure(param_os::RBParamOnlineStructure,args...)
  get_online_structure(param_os)(args...)
end

function get_on_structure(
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

function get_on_structure(
  param_os::RBParamOnlineStructure{<:RBLiftVariable},
  args...)

  -get_online_structure(param_os)(args...)
end

function eval_on_structure(
  param_os::RBParamOnlineStructure,
  sym::Symbol,
  args...)

  get_id(param_os) == sym ? get_on_structure(param_os,args...) : return
end

function eval_on_structure(
  param_os::NTuple{N,RBParamOnlineStructure},
  sym::Symbol,
  args...) where N

  ids = get_id.(param_os)
  idx = findall(x -> x == sym,ids)
  !isempty(idx) ? eval_on_structure(param_os[first(idx)],sym,args...) : return
end

function eval_on_structure(
  param_os::NTuple{N1,RBParamOnlineStructure},
  syms::NTuple{N2,Symbol},
  args...) where {N1,N2}

  all_param_os = Broadcasting(sym -> eval_on_structure(param_os,sym,args...))(syms)
  filter(!isnothing,all_param_os)
end
