struct CollectSolutionMap
  f::Function

  function CollectSolutionMap(solver::ODESolver,op::ParamTransientFEOperator)
    t0,tF = solver.t0,solver.tF
    uh0 = solver.uh0
    sol_μ = μ -> ParamTransientFESolution(solver,op,μ,uh0(μ),t0,tF)
    new(sol_μ)
  end
end

function Arrays.evaluate!(cache,k::CollectSolutionMap,μ::AbstractArray)
  sol_μ = k.f(μ)
  l = length(sol_μ)
  T = return_type(sol_μ)
  vT = Vector{T}(undef,l)
  for (n,sol_μn) in enumerate(sol_μ)
    uhn,_ = sol_μn
    vT[n] = uhn
  end
  vT
end

# function lazy_map(::typeof(evaluate),T::Type,k::AbstractArray,f::AbstractArray...)
#   s = _common_size(k,f...)
#   N = length(s)
#   LazyArray(T,Val(N),k,f...)
# end
