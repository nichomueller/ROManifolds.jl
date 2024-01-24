function get_method_operator(
  fesolver::ODESolver,
  feop::TransientPFEOperator,
  sols::ParamArray,
  params::Table)

  dtθ = fesolver.θ == 0.0 ? fesolver.dt : fesolver.dt*fesolver.θ
  times = get_stencil_times(fesolver)
  ode_cache = allocate_cache(feop,params,times)
  ode_cache = update_cache!(ode_cache,feop,params,times)
  sols_cache = zero(sols)
  get_method_operator(feop,params,times,dtθ,sols,ode_cache,sols_cache)
end

Algebra.symbolic_setup(s::BackslashSolver,mat::ParamArray) = symbolic_setup(s,testitem(mat))

Algebra.symbolic_setup(s::BackslashSolver,mat::AbstractArray{<:ParamArray}) = symbolic_setup(s,testitem(mat))

Algebra.symbolic_setup(s::LUSolver,mat::ParamArray) = symbolic_setup(s,testitem(mat))

Algebra.symbolic_setup(s::LUSolver,mat::AbstractArray{<:ParamArray}) = symbolic_setup(s,testitem(mat))

function Algebra._check_convergence(nls,b::ParamArray,m0)
  m = maximum(abs,b)
  return all(m .< nls.tol * m0)
end
