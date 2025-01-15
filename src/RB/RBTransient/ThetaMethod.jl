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
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  args...)

  dt,θ = solver.dt,solver.θ
  shift!(r,dt*(θ-1))

  paramcache = allocate_paramcache(op.op,r,us;evaluated=true)

  A = allocate_jacobian(op.op,r,us,paramcache)
  coeffA,Â = allocate_hypred_cache(op.lhs,r,args...)
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
  rmul!(b̂,-1)
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

  rbcache_lin = allocate_rbcache(solver,lop,r,us,SplitJacobian())
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
  uθ = copy(ŷ)
  dut = copy(ŷ)

  function us(u::RBParamVector)
    inv_project!(u.fe_data,trial,u.data)
    copyto!(uθ,u)
    shift!(uθ,trial,x0,θ,1-θ)
    copyto!(dut,u)
    shift!(dut,trial,x0,1/dt,-1/dt)
    (uθ,dut)
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

# utils

function ParamDataStructures.shift!(
  a::ConsecutiveParamVector,
  a0::ConsecutiveParamVector,
  α::Number,
  β::Number)

  data = get_all_data(a)
  data0 = get_all_data(a0)
  data′ = copy(data)
  np = param_length(a0)
  for ipt = param_eachindex(a)
    it = slow_index(ipt,np)
    if it == 1
      for is in axes(data,1)
        data[is,ipt] = α*data[is,ipt] + β*data0[is,ipt]
      end
    else
      for is in axes(data,1)
        data[is,ipt] = α*data[is,ipt] + β*data′[is,ipt-np]
      end
    end
  end
end

function ParamDataStructures.shift!(
  a::BlockParamVector,
  a0::BlockParamVector,
  α::Number,
  β::Number)

  @inbounds for (ai,a0i) in zip(blocks(a),blocks(a0))
    ParamDataStructures.shift!(ai,a0i,α,β)
  end
end

function ParamDataStructures.shift!(a::RBParamVector,trial::RBSpace,args...)
  ParamDataStructures.shift!(a.fe_data,args...)
  RBSteady.project!(a.data,trial,a.fe_data)
end
