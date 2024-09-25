struct LinearNonlinearRBSolver <: ParamStageOperator
  odeop::ODEOperator{NonlinearParamODE}
  odeopcache
  rx::TransientRealization
  usx::Function
  ws::Tuple{Vararg{Real}}
  lop::LinearParamStageOperator
  linear_caches
end

function LinearNonlinearRBSolver(
  odeop::ODEOperator{<:LinearNonlinearParamODE},
  odeopcache,
  rx::TransientRealization,
  usx::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  linear_caches...)

  lop = get_linear_operator(op)
  nlop = get_nonlinear_operator(op)
  lstageop = LinearParamStageOperator(lop,odeopcache,rx,usx,ws,linear_caches...)
  LinearNonlinearRBSolver(nlop,odeopcache,rx,usx,ws,lstageop,linear_caches)
end

function Algebra.allocate_residual(
  nlop::LinearNonlinearRBSolver,
  x::AbstractVector)

  odeop,odeopcache = nlop.odeop,nlop.odeopcache
  rx = nlop.rx
  usx = nlop.usx(x)
  allocate_residual(odeop,rx,usx,odeopcache)
end

function Algebra.residual!(
  nlb::Tuple,
  nlop::LinearNonlinearRBSolver,
  x::AbstractVector)

  Alin = nlop.lop.A
  blin = nlop.lop.b
  _,blincache,_... = nlop.linear_caches

  odeop,odeopcache = nlop.odeop,nlop.odeopcache
  rx = nlop.rx
  usx = nlop.usx(x)
  bnlin = residual!(nlb,odeop,rx,usx,odeopcache,blincache)

  @. bnlin = bnlin + blin + Alin*x
  return bnlin
end

function Algebra.allocate_jacobian(
  nlop::LinearNonlinearRBSolver,
  x::AbstractVector)

  odeop,odeopcache = nlop.odeop,nlop.odeopcache
  rx = nlop.rx
  usx = nlop.usx(x)
  allocate_jacobian(odeop,rx,usx,odeopcache)
end

function Algebra.jacobian!(
  nlA::Tuple,
  nlop::LinearNonlinearRBSolver,
  x::AbstractVector)

  Alin = nlop.lop.A
  Alincache,_... = nlop.linear_caches

  odeop,odeopcache = nlop.odeop,nlop.odeopcache
  rx = nlop.rx
  usx = nlop.usx(x)
  ws = nlop.ws
  Anlin = jacobian!(nlA,odeop,rx,usx,ws,odeopcache,Alincache)

  @. Anlin = Anlin + Alin
  return Anlin
end
