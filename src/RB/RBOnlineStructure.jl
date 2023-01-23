#= abstract type RBOnlineStructure{Top,Tst} end

mutable struct RBOnlineLinStructure{Top,Tst} <: RBOnlineStructure{Top<:RBLinVariable,Tst}
  op::Top
  on_structure::Tst
end

mutable struct RBOnlineBilinStructure{Top,Tst} <: RBOnlineStructure{Top<:RBBilinVariable,Tst}
  op::Top
  on_structure::Tst
end

mutable struct RBOnlineLiftStructure{Top,Tst} <: RBOnlineStructure{Top<:RBLiftVariable,Tst}
  op::Top
  on_structure::Tst
end =#

mutable struct RBOnlineStructure{Top,Tst}
  op::Top
  on_structure::Tst
end

function RBOnlineStructure(
  op::RBLinVariable{Top},
  on_structure::Tst) where {Top,Tst}

  RBOnlineStructure{RBLinVariable{Top},Tst}(op,on_structure)
end

function RBOnlineStructure(
  op::RBBilinVariable{Top,Ttr},
  on_structure::Tst) where {Top,Ttr,Tst}

  RBOnlineStructure{RBBilinVariable{Top,Ttr},Tst}(op,on_structure)
end

function RBOnlineStructure(
  op::RBLiftVariable{Top,Ttr},
  on_structure::Tst) where {Top,Ttr,Tst}

  RBOnlineStructure{RBLiftVariable{Top,Ttr},Tst}(op,on_structure)
end

function RBOnlineStructure(
  op::RBSteadyVariable,
  basis::Matrix{Float},
  coeff::Vector{Float})

  nr = get_nrows(op)
  on_structure = basis_by_coeff_mult(basis,coeff,nr)
  RBOnlineStructure(op,on_structure)
end

function RBOnlineStructure(
  op::RBSteadyVariable{Nonlinear,<:ParamTrialFESpace},
  basis::Matrix{Float},
  coeff::Function)

  nr = get_nrows(op)
  on_structure(u) = basis_by_coeff_mult(basis,coeff(u),nr)
  RBOnlineStructure(op,on_structure)
end

function RBOnlineStructure(
  op::RBUnsteadyVariable,
  basis::Matrix{Float},
  coeff::Matrix{Float})

  btbtc = coeff_by_time_bases(op,coeff)
  ns_row = get_ns(get_rbspace_row(op))
  basis_block = blocks(basis,ns_row)

  nr = get_nrows(op)
  on_structure = basis_by_coeff_mult(basis_block,btbtc,nr)
  RBOnlineStructure(op,on_structure)
end

function RBOnlineStructure(
  op::RBUnsteadyBilinVariable{Nonlinear,<:ParamTransientTrialFESpace},
  basis::Matrix{Float},
  coeff::Function)

  btbtc = coeff_by_time_bases(op,coeff)
  ns_row = get_ns(get_rbspace_row(op))
  basis_block = blocks(basis,ns_row)

  nr = get_nrows(op)
  mat(u) = basis_by_coeff_mult(basis_block,btbtc[1](u),nr)
  mat_shift(u) = basis_by_coeff_mult(basis_block,btbtc[2](u),nr)
  RBOnlineStructure(op,(mat,mat_shift))
end

get_op(rbos::RBOnlineStructure) = rbos.op
get_id(rbos::RBOnlineStructure) = get_id(get_op(rbos))

function get_op(
  rbos::RBOnlineStructure,
  sym::Symbol)

  get_id(rbos) == sym ? rbos.op : return
end

function get_op(
  rbos::NTuple{N,RBOnlineStructure},
  sym::Symbol) where N

  all_rbos = Broadcasting(os -> get_op(os,sym))(rbos)
  filter(!isnothing,all_rbos)
end

function get_op(
  rbos::NTuple{N1,RBOnlineStructure},
  syms::NTuple{N2,Symbol}) where {N1,N2}

  all_rbos = Broadcasting(sym -> get_op(rbos,sym))(syms)
  filter(!isnothing,all_rbos)
end

get_online_structure(rbos::RBOnlineStructure) = rbos.on_structure

function eval_on_structure(rbos::RBOnlineStructure)
  get_online_structure(rbos)
end

function eval_on_structure(
  rbos::RBOnlineStructure{RBUnsteadyBilinVariable,NTuple{2,Matrix{Float}}})

  op = get_op(rbos)
  θ = get_θ(op)
  dt = get_dt(op)
  mat,mat_shift = get_online_structure(rbos)

  if get_id(op) == :M
    mat/dt - mat_shift/dt
  else
    θ*mat + (1-θ)*mat_shift
  end
end

function eval_on_structure(
  rbos::RBOnlineStructure{RBUnsteadyBilinVariable,NTuple{2,Function}})

  op = get_op(rbos)
  θ = get_θ(op)
  mat,mat_shift = get_online_structure(rbos)
  u -> θ*mat(u) + (1-θ)*mat_shift(u)
end

function eval_on_structure(
  rbos::RBOnlineStructure{RBLiftVariable,Matrix{Float}})
  -get_online_structure(rbos)
end

function eval_on_structure(
  rbos::RBOnlineStructure{RBLiftVariable,Function})

  u -> -get_online_structure(rbos)(u)
end

function eval_on_structure(
  rbos::RBOnlineStructure,
  sym::Symbol)

  get_id(rbos) == sym ? eval_on_structure(rbos) : return
end

function eval_on_structure(
  rbos::NTuple{N,RBOnlineStructure},
  sym::Symbol) where N

  ids = get_id.(rbos)
  idx = findall(x -> x == sym,ids)
  !isempty(idx) ? eval_on_structure(rbos[first(idx)],sym) : return
end

function eval_on_structure(
  rbos::NTuple{N1,RBOnlineStructure},
  syms::NTuple{N2,Symbol}) where {N1,N2}

  all_rbos = Broadcasting(sym -> eval_on_structure(rbos,sym))(syms)
  filter(!isnothing,all_rbos)
end
