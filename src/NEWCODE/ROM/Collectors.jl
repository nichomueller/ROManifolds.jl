_get_nsnaps(args...) = 1

_get_nsnaps(::Union{TimeAffinity,NonAffinity},params::Table) = length(params)

function collect_solutions(
  op::ParamTransientFEOperator,
  solver::ODESolver,
  params::Table;
  kwargs...)

  aff = NonAffinity()
  nsol = _get_nsnaps(aff,params)
  printstyled("Generating $nsol solution snapshots\n";color=:blue)

  s = allocate_solution(op.test,solver)
  sol = solutions(s,op,solver,params)
  TransientSnapshots(aff,sol,nsol;kwargs...)
end

function collect_residuals(
  op::ParamTransientFEOperator,
  solver::ODESolver,
  sols::TransientSnapshots,
  params::Table,
  args...)

  aff = affinity_residual(op,solver,params,args...)
  nres = _get_nsnaps(aff,params)

  printstyled("Generating $nres residual snapshots, affinity: $aff\n";color=:blue)

  res_iter = init_vec_iterator(op,solver,args...)
  res = residuals(aff,op,solver,res_iter,sols,params)
  TransientSnapshots(aff,res,nres)
end

function collect_jacobians(
  op::ParamTransientFEOperator,
  solver::ODESolver,
  sols::TransientSnapshots,
  params::Table,
  args...)

  aff = affinity_jacobian(op,solver,params,args...)
  njac = _get_nsnaps(aff,params)

  printstyled("Generating $njac jacobian snapshots, affinity: $aff\n";color=:blue)

  jac_iter = init_mat_iterator(op,solver,args...)
  jac = jacobians(aff,op,solver,jac_iter,sols,params)
  TransientSnapshots(aff,jac,njac)
end

function collect_djacobians(
  op::ParamTransientFEOperator,
  solver::ODESolver,
  sols::TransientSnapshots,
  params::Table,
  args...)

  aff = affinity_jacobian(op,solver,params,args...;i=2)
  ndjac = _get_nsnaps(aff,params)

  printstyled("Generating $ndjac djacobian/dt snapshots, affinity: $aff\n";color=:blue)

  djac_iter = init_mat_iterator(op,solver,args...;i=2)
  djac = djacobians(aff,op,solver,djac_iter,sols,params)
  TransientSnapshots(aff,djac,ndjac)
end

function allocate_solution(test::SingleFieldFESpace,solver::ODESolver)
  space_ndofs = num_free_dofs(test)
  time_ndofs = get_time_ndofs(solver)
  cache = fill(0.,space_ndofs,time_ndofs)
  cache
end

function allocate_solution(test::MultiFieldFESpace,args...)
  map(t->allocate_solution(t,args...),test.spaces)
end

function solutions(
  cache,
  op::ParamTransientFEOperator,
  solver::ODESolver,
  params::Table)

  pmap(params) do μ
    solution!(cache,op,solver,μ)
  end
end

function solution!(
  cache,
  op::ParamTransientFEOperator,
  solver::ODESolver,
  μ::AbstractVector)

  sol = solve(op,solver,μ,solver.uh0(μ))
  n = 1
  if isa(cache,AbstractMatrix)
    for soln in sol
      setindex!(cache,soln,:,n)
      n += 1
    end
  elseif isa(cache,Vector{<:AbstractMatrix})
    for soln in sol
      map((cache,sol) -> setindex!(cache,sol,:,n),cache,soln)
      n += 1
    end
  else
    @unreachable
  end
  cache
end

abstract type IterativeCollector end
abstract type TransientIterativeCollector end

mutable struct IterativeVecCollector <: IterativeCollector
  f::Function
  xh::AbstractVector
  μ::AbstractVector
  cache
end

mutable struct TransientIterativeVecCollector <: TransientIterativeCollector
  f::Function
  xh::Tuple{Vararg{AbstractVector}}
  μ::AbstractVector
  t::Float
  cache
end

mutable struct IterativeMatCollector <: IterativeCollector
  f::Function
  xh::AbstractVector
  μ::AbstractVector
  cache
end

mutable struct TransientIterativeMatCollector <: TransientIterativeCollector
  f::Function
  xh::Tuple{Vararg{AbstractVector}}
  μ::AbstractVector
  t::Float
  cache
end

function init_vec_iterator(
  op::ParamTransientFEOperator,
  ::θMethod,
  trian::Triangulation,
  filter::Tuple{Vararg{Int}},
  args...)

  row,_ = filter
  test_row = get_test(op)[row]
  dv_row = _get_fe_basis(op.test,row)
  assem_row = SparseMatrixAssembler(test_row,test_row)
  b = allocate_residual(op;assem=assem_row)

  function f(
    xh::Tuple{Vararg{AbstractVector}},
    μ::AbstractVector,
    t::Real,
    cache)

    r = copy(b)

    u = evaluation_function(op,xh,cache)
    vecdata = collect_cell_vector(test_row,op.res(μ,t,u,dv_row,args...),trian)
    assemble_vector_add!(r,assem_row,vecdata)
    r .*= -1.0
  end

  μ = realization(op)
  t = 0.
  xhθ = zeros(num_free_dofs(op.test))
  xh0 = copy(xhθ)
  cache = allocate_cache(op)

  TransientIterativeVecCollector(f,(xhθ,xh0),μ,t,cache)
end

function init_mat_iterator(
  op::ParamTransientFEOperator,
  solver::θMethod,
  trian::Triangulation,
  filter::Tuple{Vararg{Int}},
  args...;
  i::Int=1)

  row,col = filter
  test_row = op.test[row]
  _trial = get_trial(op)(nothing,nothing)
  _trial_col = _trial[col]
  dv_row = _get_fe_basis(op.test,row)
  du_col = _get_trial_fe_basis(_trial,col)
  assem_row_col = SparseMatrixAssembler(_trial_col,test_row)
  A = allocate_jacobian(op;assem=assem_row_col)
  annz = compress(A)
  γ = (1.0,1/(solver.dt*solver.θ))

  function f(
    xh::Tuple{Vararg{AbstractVector}},
    μ::AbstractVector,
    t::Float,
    cache)

    J = copy(A)
    jnnz = copy(annz)

    trial, = cache[1]
    trial_col = trial[col]
    u = evaluation_function(op,xh,cache)
    u_col = filter_evaluation_function(u,col)
    matdata = collect_cell_matrix(trial_col,test_row,
      γ[i]*op.jacs[i](μ,t,u_col,du_col,dv_row,args...),trian)
    J = assemble_matrix_add!(J,assem_row_col,matdata)
    compress!(jnnz,J)
    jnnz
  end

  μ = realization(op)
  t = 0.
  xhθ = zeros(num_free_dofs(op.test))
  xh0 = copy(xhθ)
  cache = allocate_cache(op)
  TransientIterativeMatCollector(f,(xhθ,xh0),μ,t,cache)
end

function update!(
  itc::TransientIterativeCollector,
  op::ParamTransientFEOperator,
  solver::θMethod,
  μ::AbstractVector,
  t::Real,
  xhθ=get_free_dof_values(solver.uh0(μ)))

  itc.μ = μ
  itc.t = t
  xh0 = copy(xhθ)
  itc.xh = (xhθ,xh0)
  cache = itc.cache
  update_cache!(cache,op,μ,t)
  itc.cache = cache
  return
end

function evaluate!(itc::IterativeVecCollector)::Vector{Float}
  itc.f(itc.xh,itc.μ,itc.cache)
end

function evaluate!(itc::IterativeMatCollector)::NnzArray{SparseMatrixCSC}
  itc.f(itc.xh,itc.μ,itc.cache)
end

function evaluate!(itc::TransientIterativeVecCollector)::Vector{Float}
  itc.f(itc.xh,itc.μ,itc.t,itc.cache)
end

function evaluate!(itc::TransientIterativeMatCollector)::NnzArray{SparseMatrixCSC}
  itc.f(itc.xh,itc.μ,itc.t,itc.cache)
end

for fun in (:residuals,:jacobians,:djacobians)
  @eval begin
    function $fun(
      ::Union{ZeroAffinity,ParamTimeAffinity},
      op::ParamTransientFEOperator,
      solver::θMethod,
      itc::TransientIterativeCollector,
      ::TransientSnapshots,
      params::Table)

      μ = first(params)
      t = first(get_times(solver))
      update!(itc,op,solver,μ,t)
      evaluate!(itc)
    end

    function $fun(
      ::ParamAffinity,
      op::ParamTransientFEOperator,
      solver::θMethod,
      itc::TransientIterativeCollector,
      ::TransientSnapshots,
      params::Table)

      times = get_times(solver)
      μ = first(params)
      map(times) do t
        update!(itc,op,solver,μ,t)
        evaluate!(itc)
      end
    end

    function $fun(
      ::TimeAffinity,
      op::ParamTransientFEOperator,
      solver::θMethod,
      itc::TransientIterativeCollector,
      ::TransientSnapshots,
      params::Table)

      t = first(get_times(solver))
      pmap(enumerate(params)) do (nμ,μ)
        update!(itc,op,solver,μ,t)
        evaluate!(itc)
      end
    end

    function $fun(
      ::NonAffinity,
      op::ParamTransientFEOperator{Affine},
      solver::θMethod,
      itc::TransientIterativeCollector,
      ::TransientSnapshots,
      params::Table)

      times = get_times(solver)
      tdofs = length(times)
      pmap(enumerate(params)) do (nμ,μ)
        qt = map(times) do t
          update!(itc,op,solver,μ,t)
          evaluate!(itc)
        end
        hcat(qt...)
      end
    end

    function $fun(
      ::NonAffinity,
      op::ParamTransientFEOperator,
      solver::θMethod,
      itc::TransientIterativeCollector,
      s::TransientSnapshots,
      params::Table)

      times = get_times(solver)
      tdofs = length(times)
      sols = get_datum(s)

      pmap(enumerate(params)) do (nμ,μ)
        function _snaps_μ_tθ(sols,μ,nμ)
          θ = solver.θ
          ic = solver.uh0(μ)
          ich = get_free_dof_values(ic)
          idx_nμ = (nμ-1)*tdofs+1:nμ*tdofs
          prev_sols = hcat(ich,sols[:,idx_nμ[1:end-1]])
          θ*sols[:,idx_nμ] + (1-θ)*prev_sols
        end

        ic = solver.uh0(μ)
        ich = get_free_dof_values(ic)
        idx_nμ = (nμ-1)*tdofs+1:nμ*tdofs
        sols_μθ = _snaps_μ_tθ(sols,μ,nμ)
        qt = map(enumerate(times)) do (nt,t)
          xhθ = sols_μθ[:,nt]
          update!(itc,op,solver,μ,t,xhθ)
          evaluate!(itc)
        end
        hcat(qt...)
      end
    end
  end
end
