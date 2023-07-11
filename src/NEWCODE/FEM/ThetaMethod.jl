function solve_step!(
  uf::AbstractVector,
  op::ParamTransientFEOperator,
  solver::θMethod,
  μ::AbstractVector,
  u0::AbstractVector,
  t0::Real,
  cache)

  dt = solver.dt
  solver.θ == 0.0 ? dtθ = dt : dtθ = dt*solver.θ
  tθ = t0+dtθ

  if isnothing(cache)
    ode_cache = allocate_cache(op)
    vθ = similar(u0)
    nl_cache = nothing
  else
    ode_cache,vθ,nl_cache = cache
  end

  ode_cache = update_cache!(ode_cache,op,μ,tθ)

  nlop = ParamThetaMethodNonlinearOperator(op,μ,tθ,dtθ,u0,ode_cache,vθ)

  nl_cache = solve!(uf,solver.nls,nlop,nl_cache)

  if 0.0 < solver.θ < 1.0
    uf = uf*(1.0/solver.θ)-u0*((1-solver.θ)/solver.θ)
  end

  cache = (ode_cache,vθ,nl_cache)

  tf = t0+dt
  return (uf,tf,cache)
end

"""
Nonlinear operator that represents the θ-method nonlinear operator at a
given time step, i.e., A(t,u_n+θ,(u_n+θ-u_n)/dt)
"""
struct ParamThetaMethodNonlinearOperator <: NonlinearOperator
  odeop::ParamTransientFEOperator
  μ::AbstractVector
  tθ::Float64
  dtθ::Float64
  u0::AbstractVector
  ode_cache
  vθ::AbstractVector
end

function residual!(
  b::AbstractVector,
  op::ParamThetaMethodNonlinearOperator,
  x::AbstractVector)

  uθ = x
  vθ = (x-op.u0)/op.dtθ
  residual!(b,op.odeop,op.μ,op.tθ,(uθ,vθ),op.ode_cache)
end

function jacobian!(
  A::AbstractMatrix,
  op::ParamThetaMethodNonlinearOperator,
  x::AbstractVector)

  uF = x
  vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(A))
  LinearAlgebra.fillstored!(A,z)
  jacobians!(A,op.odeop,op.μ,op.tθ,(uF,vθ),(1.0,1/op.dtθ),op.ode_cache)
end

function allocate_residual(
  op::ParamThetaMethodNonlinearOperator,
  x::AbstractVector)

  allocate_residual(op.odeop,x,op.ode_cache)
end

function allocate_jacobian(
  op::ParamThetaMethodNonlinearOperator,
  x::AbstractVector)

  allocate_jacobian(op.odeop,x,op.ode_cache)
end

filter_evaluation_function(u,args...) = u

function filter_evaluation_function(
  u::Gridap.ODEs.TransientFETools.TransientMultiFieldCellField,
  col::Int)

  u_col = Any[]
  for nf = eachindex(u.transient_single_fields)
    nf == col ? push!(u_col,u[col]) : push!(u_col,nothing)
  end
  u_col
end

# function _vecdata_residual(
#   op::ParamTransientFEOperator,
#   solver::θMethod,
#   sols::AbstractArray,
#   params::AbstractArray,
#   filter::Tuple{Vararg{Int}},
#   args...;
#   trian::Triangulation=get_triangulation(op.test))

#   row,_ = filter
#   test_row = get_test(op)[row]
#   trial = get_trial(op)
#   dv_row = _get_fe_basis(op.test,row)
#   u = _evaluation_function(solver,trial,sols,params)
#   assem_row = SparseMatrixAssembler(test_row,test_row)
#   op.assem = assem_row
#   (μ,t) -> collect_cell_vector(test_row,op.res(μ,t,u(μ,t),dv_row,args...),trian)
# end

# function _matdata_jacobian(
#   op::ParamTransientFEOperator,
#   solver::θMethod,
#   sols::AbstractArray,
#   params::AbstractArray,
#   filter::Tuple{Vararg{Int}},
#   args...;
#   trian::Triangulation=get_triangulation(op.test))

#   row,col = filter
#   test_row = get_test(op)[row]
#   trial_col = get_trial(op)[col]
#   dv_row = _get_fe_basis(op.test,row)
#   du_col = _get_trial_fe_basis(get_trial(op)(nothing,nothing),col)
#   u = _evaluation_function(solver,trial,sols,params)
#   assem_row_col = SparseMatrixAssembler(trial_col(nothing,nothing)[col],test_row)
#   op.assem = assem_row_col

#   γ = (1.0,1/(solver.dt*solver.θ))
#   function matdata(μ,t)
#     u_col(μ,t) = filter_evaluation_function(u(μ,t),col)
#     _matdata = ()
#     for (i,γᵢ) in enumerate(γ)
#       if γᵢ > 0.0
#         _matdata = (_matdata...,
#           collect_cell_matrix(
#           trial_col(μ,t),
#           test_row,
#           γᵢ*op.jacs[i](μ,t,u_col(μ,t),dv_row,du_col,args...),
#           trian))
#       end
#     end
#     _vcat_matdata(_matdata)
#   end

#   matdata
# end
