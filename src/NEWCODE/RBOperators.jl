# root = pwd()
# using MPI,MPIClusterManagers,Distributed
# include("$root/src/FEM/FEM.jl")
# include("$root/src/RBTests/RBTests.jl")

# mesh = "elasticity_3cyl.json"
# test_path = "$root/tests/poisson/unsteady/$mesh"
# bnd_info = Dict("dirichlet" => ["dirichlet"],"neumann" => ["neumann"])
# order = 1
# degree = 2

# fepath = fem_path(test_path)
# mshpath = mesh_path(test_path,mesh)
# model = get_discrete_model(mshpath,bnd_info)
# Ω = Triangulation(model)
# dΩ = Measure(Ω,degree)

# g(x,t::Real) = x[1]
# g(t::Real) = x->g(x,t)

# reffe = Gridap.ReferenceFE(lagrangian,Float,1)
# test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
# trial = TransientTrialFESpace(test,g)
# lhs_t(u,v) = ∫(v*u)dΩ
# du = get_trial_fe_basis(trial(0.1))
# dv = get_fe_basis(test)

# dc = lhs_t(du,dv)
# propertynames(dc.dict)

# fun(x) = 3*x[1]+x[2]
# nfree = test.nfree
# f = GenericField(fun)
# fc = Fill(f,num_cells(Ω))
# fΩ = GenericCellField(fc,Ω,PhysicalDomain())
# ∫(fΩ*dv*du)dΩ

# @which integrate(∫(fΩ).object,dΩ)

# free_values = rand(num_cells(Ω))
# dirichlet_values = get_dirichlet_dof_values(test(0))
# cell_vals = scatter_free_and_dirichlet_values(test(0),free_values,dirichlet_values)
# cell_field = CellField(test(0),cell_vals)
# fΩ1 = SingleFieldFEFunction(cell_field,cell_vals,free_values,dirichlet_values,test(0))
# GenericCellField(fΩ1,Ω,PhysicalDomain())

function reduce_fe_operator(
  feop::ParamFEOperator,
  rbspace::RBBasis;
  kwargs...)

  μ = realization(feop.pspace)
  du = get_trial_fe_basis(feop.trial(μ))
  dv = get_fe_basis(feop.test)
  jac_affinity = isaffine(feop.jac,μ,du,dv)
  rb_jac =
  res_affinity = isaffine(feop.res,μ,dv)
end

function reduce_fe_operator(
  feop::ParamTransientFEOperator,
  rbspace::RBBasis;
  kwargs...)

  syms = intersect(propertynames(feop),(:jac,:jacs,:res))
  μ = realization(feop.pspace)
  du = get_trial_fe_basis(feop.trial(μ))
  dv = get_fe_basis(feop.test)
  rb_terms = map(
    sym->reduce_fe_operator(getproperty(feop,sym),feop,rbspace;kwargs...),
    syms)
  RBOperator(rb_terms...)
end

function reduce_fe_operator(
  terms::Tuple{Vararg{Function}},
  feop,
  rbspace::RBBasis;
  kwargs...)

  map(term->reduce_fe_term(term,feop,rbspace),terms;kwargs...)
end

function reduce_fe_term(
  term::Function,
  feop,
  rbspace::RBBasis;
  kwargs...)

  affinity = isaffine(term,feop)
  reduce_fe_term(Val{affinity}(),term,feop,rbspace;kwargs...)
end

function reduce_fe_term(
  ::Val{true},
  term::Function,
  feop,
  rbspace::RBBasis;
  kwargs...)

end

function reduce_fe_term(
  ::Val{false},
  term::Function,
  feop,
  rbspace::RBBasis;
  kwargs...)

end

function isaffine(term::Function,args...)
  try term(args...)
    Val{false}()
  catch
    Val{true}()
  end
end
