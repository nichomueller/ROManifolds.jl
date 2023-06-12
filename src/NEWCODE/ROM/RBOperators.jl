abstract type RBIntegrationDomain end

struct AffineDecomposition
  basis::RBSpace
  rb_interpolation::LU
  rb_integration::RBIntegrationDomain
end

abstract type RBOperator end

struct RBSteadyOperator <: RBOperator
  jac::AffineDecomposition
  res::AffineDecomposition
end

struct RBUnsteadyOperator <: RBOperator
  jac::Tuple{Vararg{AffineDecomposition}}
  res::AffineDecomposition
end

function reduce_fe_operator(
  feop::ParamFEOperator,
  rbspace::RBSpace;
  kwargs...)

  μ = realization(feop.pspace)
  du = get_trial_fe_basis(feop.trial(μ))
  dv = get_fe_basis(feop.test)
  reduce_fe_operator(feop,rbspace,μ,du,dv;kwargs...)
end

function reduce_fe_operator(
  feop::ParamTransientFEOperator,
  rbspace::RBSpace;
  kwargs...)

  μ = realization(feop.pspace)
  t = realization(feop.tinfo)
  du = get_trial_fe_basis(feop.trial(μ,t))
  dv = get_fe_basis(feop.test)
  reduce_fe_operator(feop,rbspace,(μ,t),du,dv;kwargs...)
end

function reduce_fe_operator(
  feop,
  rbspace::RBSpace,
  input,
  du::FEBasis,
  dv::FEBasis;
  kwargs...)

  jac_affinity = isaffine(feop.jac,input,du,dv)
  jac_solver = get_bilinear_solver(jac_affinity)
  rb_jac = reduce_fe_term(jac_affinity,jac_solver,feop,rbspace;kwargs...)
  res_affinity = isaffine(feop.res,input,dv)
  res_solver = get_linear_solver(res_affinity)
  rb_res = reduce_fe_term(res_affinity,res_solver,feop,rbspace;kwargs...)
  RBOperator(rb_jac,rb_res)
end

function reduce_fe_term(
  ::Val{true},
  solver,
  feop,
  rbspace::RBSpace;
  kwargs...)

  affine_vec = assemble_vector(feop)
  basis,rb_interp,rb_integr = cache_mdeim(feop,rbspace)
  reduce!(basis,affine_vec,rbspace)
  AffineDecomposition(basis,rb_interp,rb_integr)
end

function reduce_fe_term(
  ::Val{false},
  solver,
  feop,
  rbspace::RBSpace;
  n_snaps=20,kwargs...)

  snaps = generate_snapshots(feop,solver,n_snaps)
  affine_vecs = reduce(snaps;kwargs...)
  rb_integration_temp = get_integration_domain(affine_vecs)
  rb_interpolation = get_mdeim_interpolation(affine_vecs,rb_integration_temp)
  basis = copy(affine_vecs)
  reduce!(basis,affine_vecs,rbspace;kwargs...)
  recast_integration_domain!(rb_integration)
  AffineDecomposition(basis,rb_interpolation,rb_integration)
end

function isaffine(terms::Tuple{Vararg{Function}},args...)
  map(term->isaffine(term,args...),terms)
end

function isaffine(term::Function,args...)
  try term(args...)
    typeof(term(args...)) <: DomainContribution ? Val{false}() : Val{true}()
  catch
    Val{true}()
  end
end

get_linear_solver(::Val{true}) = assemble_vector

get_bilinear_solver(::Val{true}) = assemble_matrix

get_linear_solver(::Val{false}) = LinearSnapshots()

get_bilinear_solver(::Val{false}) = BilinearSnapshots()

function collect_snapshot!(cache,data::Tuple{Vararg{Any}},filters)
  sol_cache,param_cache = cache

  printstyled("Computing snapshot $(sol.k)\n";color=:blue)
  if isa(sol_cache,NnzMatrix)
    copyto!(sol_cache,sol.uh)
  else
    map((cache,sol) -> copyto!(cache,sol),sol_cache,sol.uh)
  end
  copyto!(param_cache,sol.μ)
  printstyled("Successfully computed snapshot $(sol.k)\n";color=:blue)

  sol_cache,param_cache
end
