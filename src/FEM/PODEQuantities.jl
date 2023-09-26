abstract type PODESolver <: ODESolver end

struct PThetaMethod <: PODESolver
  nls::NonlinearSolver
  uh0::Function
  θ::Float
  dt::Float
  t0::Real
  tf::Real
end

function get_times(fesolver::PThetaMethod)
  θ = fesolver.θ
  dt = fesolver.dt
  t0 = fesolver.t0
  tf = fesolver.tf
  collect(t0:dt:tf-dt) .+ dt*θ
end

struct PODESolution{T}
  solver::PODESolver
  op::PODEOperator
  μ::AbstractVector
  u0::PTArray
  t0::Real
  tf::Real

  function PODESolution(
    solver::PODESolver,
    op::PODEOperator,
    μ::AbstractVector,
    u0::PTArray,
    t0::Real,
    tf::Real)

    test = get_test(op.feop)
    T = typeof(test)
    new{T}(solver,op,μ,u0,t0,tf)
  end
end

function Base.iterate(sol::PODESolution)
  uf = copy(sol.u0)
  u0 = copy(sol.u0)
  t0 = sol.t0
  n = 0
  cache = nothing

  uf,tf,cache = solution_step!(uf,sol.solver,sol.op,sol.μ,u0,t0,cache)

  u0 .= uf
  n += 1
  state = (uf,u0,tf,n,cache)

  return (uf,tf,n),state
end

function Base.iterate(sol::PODESolution,state)
  uf,u0,t0,n,cache = state

  if t0 >= sol.tf - 100*eps()
    return nothing
  end

  uf,tf,cache = solution_step!(uf,sol.solver,sol.op,sol.μ,u0,t0,cache)

  u0 .= uf
  n += 1
  state = (uf,u0,tf,n,cache)

  return (uf,tf,n),state
end

for f in (:allocate_solution,:get_solution)
  @eval begin
    function $f(op::PODEOperator,args...)
      $f(get_test(op.feop),args...)
    end
  end
end

function allocate_solution(fe::FESpace,niter::Int)
  T = get_vector_type(fe)
  Vector{PTArray{T}}(undef,niter)
end

function allocate_solution(fe::MultiFieldFESpace,niter::Int)
  T = get_vector_type(fe)
  Vector{Vector{PTArray{T}}}(undef,niter)
end

function get_solution(::FESpace,free_values::PTArray)
  free_values
end

function get_solution(fe::MultiFieldFESpace,free_values::PTArray)
  blocks = map(1:length(fe.spaces)) do i
    restrict_to_field(fe,free_values,i)
  end
  PTArray.(blocks)
end
