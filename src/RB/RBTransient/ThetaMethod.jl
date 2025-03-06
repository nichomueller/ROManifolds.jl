function RBSteady.allocate_rbcache(
  solver::ThetaMethod,
  op::TransientRBOperator,
  r::TransientRealization,
  u::AbstractParamVector)

  w = copy(u)
  fill!(w,zero(eltype(w)))
  us = (w,w)
  allocate_rbcache(solver,op,r,us)
end

# linear case

function RBSteady.allocate_rbcache(
  solver::ThetaMethod,
  op::TransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  dt,θ = solver.dt,solver.θ
  shift!(r,dt*(θ-1))

  paramcache = allocate_paramcache(op.op,r;evaluated=true)

  A = allocate_jacobian(op.op,r,us,paramcache)
  coeffA,Â = allocate_hypred_cache(op.lhs,r)
  Acache = HRParamArray(A,coeffA,Â)

  b = allocate_residual(op.op,r,us,paramcache)
  coeffb,b̂ = allocate_hypred_cache(op.rhs,r)
  bcache = HRParamArray(b,coeffb,b̂)

  trial = evaluate(get_trial(op),r)

  shift!(r,dt*(1-θ))

  return RBCache(Acache,bcache,trial,paramcache)
end

function Algebra.solve!(
  x̂::AbstractVector,
  solver::ThetaMethod,
  op::TransientRBOperator,
  r::TransientRealization,
  x::AbstractVector,
  x0::AbstractVector,
  rbcache::RBCache)

  sysslvr = solver.sysslvr
  dt,θ = solver.dt,solver.θ
  fill!(x,zero(eltype(x)))
  usx = (x,x)
  ws = (1,1)

  shift!(r,dt*(θ-1))
  Â = jacobian(op,r,usx,ws,rbcache)
  b̂ = residual(op,r,usx,rbcache)
  solve!(x̂,sysslvr,Â,b̂)
  shift!(r,dt*(1-θ))
  return x̂
end

# linear - nonlinear case

function RBSteady.allocate_rbcache(
  solver::ThetaMethod,
  op::LinearNonlinearTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  ws = (1,1)

  lop = get_linear_operator(op)
  nlop = get_nonlinear_operator(op)

  rbcache_lin = allocate_rbcache(solver,lop,r,us)
  rbcache_nlin = allocate_rbcache(solver,nlop,r,us)
  A_lin = jacobian(lop,r,us,ws,rbcache_lin)
  b_lin = residual(lop,r,us,rbcache_lin)

  return LinearNonlinearRBCache(rbcache_nlin,A_lin,b_lin)
end

function Algebra.solve!(
  x̂::AbstractVector,
  solver::ThetaMethod,
  op::TransientRBOperator{LinearNonlinearParamODE},
  r::TransientRealization,
  x::AbstractVector,
  x0::AbstractVector,
  cache::LinearNonlinearRBCache)

  sysslvr = solver.sysslvr
  dt,θ = solver.dt,solver.θ
  trial = cache.rbcache.trial
  ŷ = RBParamVector(x̂,x)

  function us(u::RBParamVector)
    inv_project!(u.fe_data,trial,u.data)
    (u,u)
  end

  ws = (1,1)
  usx = us(ŷ)

  Âcache = jacobian(op,r,usx,ws,cache)
  b̂cache = residual(op,r,usx,cache)

  Â_item = testitem(Âcache)
  x̂_item = testitem(x̂)
  dx̂ = allocate_in_domain(Â_item)
  fill!(dx̂,zero(eltype(dx̂)))
  ss = symbolic_setup(BackslashSolver(),Â_item)
  ns = numerical_setup(ss,Â_item,x̂_item)

  shift!(r,dt*(θ-1))
  nlop = ParamStageOperator(op,cache,r,us,ws)
  Algebra._solve_nr!(ŷ,Âcache,b̂cache,dx̂,ns,sysslvr,nlop)
  shift!(r,dt*(1-θ))

  return x̂
end
