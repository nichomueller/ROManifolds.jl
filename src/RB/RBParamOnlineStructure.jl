mutable struct RBParamOnlineStructure{Top}
  op::Top
  on_structure::Function
end

function RBParamOnlineStructure(
  op::RBLinVariable{Top},
  on_structure::Function) where Top

  RBParamOnlineStructure{RBLinVariable{Top}}(op,on_structure)
end

function RBParamOnlineStructure(
  op::RBBilinVariable{Top,Ttr},
  on_structure::Function) where {Top,Ttr}

  RBParamOnlineStructure{RBBilinVariable{Top,Ttr}}(op,on_structure)
end

function RBParamOnlineStructure(
  op::RBLiftVariable{Top,Ttr},
  on_structure::Function) where {Top,Ttr}

  RBParamOnlineStructure{RBLiftVariable{Top,Ttr}}(op,on_structure)
end

#= function RBParamOnlineStructure(
  rb_structure::Tuple,
  args...;
  kwargs...)

  ntup_rb_structure = expand(rb_structure)
  RBParamOnlineStructure(ntup_rb_structure,args...;kwargs...)
end =#

function RBParamOnlineStructure(
  rb_structure::NTuple{N,RBOfflineStructure};
  kwargs...) where N

  Broadcasting(rb->RBParamOnlineStructure(rb;kwargs...))(rb_structure)
end

function RBParamOnlineStructure(
  rb_structure::NTuple{N,RBOfflineStructure},
  rbspaceθ::NTuple{N,RBSpace};
  kwargs...) where N

  Broadcasting((rbs,rbsθ)->RBParamOnlineStructure(rbs,rbsθ;kwargs...))(rb_structure,rbspaceθ)
end

function RBParamOnlineStructure(
  rb_structure::RBOfflineStructure,
  args...;
  kwargs...)

  op = get_op(rb_structure)
  basis = get_offline_structure(rb_structure)
  coeff = compute_coefficient(op,basis;kwargs...)
  os = eval_off_structure(rb_structure,args...)
  param_on_structure(μ::Param) = rb_online_product(os,coeff(μ);nr=get_nrows(op))

  RBParamOnlineStructure(op,param_on_structure)
end

function RBParamOnlineStructure(
  rb_structure::RBNonlinearStructure,
  args...;
  kwargs...)

  op = get_op(rb_structure)
  basis = get_offline_structure(rb_structure)
  coeff = compute_coefficient(op,basis;kwargs...)
  os = eval_off_structure(rb_structure,args...)
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

  all_param_os = Broadcasting(os -> get_op(os,sym))(param_os)
  filter(!isnothing,all_param_os)
end

function get_op(
  param_os::NTuple{N1,RBParamOnlineStructure},
  syms::NTuple{N2,Symbol}) where {N1,N2}

  all_param_os = Broadcasting(sym -> get_op(param_os,sym))(syms)
  filter(!isnothing,all_param_os)
end

get_online_structure(param_os::RBParamOnlineStructure) = param_os.on_structure

function eval_on_structure(param_os::RBParamOnlineStructure,args...)
  get_online_structure(param_os)(args...)
end

function eval_on_structure(
  param_os::RBParamOnlineStructure{RBUnsteadyBilinVariable},
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
  param_os::RBParamOnlineStructure{RBLiftVariable},
  q)

  -get_online_structure(param_os)(q)
end

function eval_on_structure(
  param_os::RBParamOnlineStructure,
  sym::Symbol,
  args...)

  get_id(param_os) == sym ? eval_on_structure(param_os,args...) : return
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
