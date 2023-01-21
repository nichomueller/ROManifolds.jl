abstract type MySpaces end

function dirichlet_dofs_on_full_trian(
  space::FESpace,
  space_no_bc::UnconstrainedFESpace)

  cell_dof_ids = get_cell_dof_ids(space)
  cell_dof_ids_no_bc = get_cell_dof_ids(space_no_bc)

  dirichlet_dofs = zeros(Int,num_dirichlet_dofs(space))
  if !isempty(dirichlet_dofs)
    for cell = eachindex(cell_dof_ids)
      for (ids,ids_no_bc) in zip(cell_dof_ids[cell],cell_dof_ids_no_bc[cell])
        if ids<0
          dirichlet_dofs[abs(ids)]=ids_no_bc
        end
      end
    end
  end

  dirichlet_dofs
end

struct MyTests <: MySpaces
  test::SingleFieldFESpace
  test_no_bc::SingleFieldFESpace
  ddofs_on_full_trian::Vector{Int}

  function MyTests(
    test::SingleFieldFESpace,
    test_no_bc::SingleFieldFESpace)

    ddofs_on_full_trian = dirichlet_dofs_on_full_trian(test,test_no_bc)
    new(test,test_no_bc,ddofs_on_full_trian)
  end
end

function MyTests(model,reffe,args...;kwargs...)
  test = TestFESpace(model,reffe,args...;kwargs...)
  test_no_bc = FESpace(model,reffe)
  MyTests(test,test_no_bc)
end

function MyTrial(test::SingleFieldFESpace)
  HomogeneousTrialFESpace(test)
end

function MyTrial(
  test::SingleFieldFESpace,
  g::Function,
  ::Val{true})
  ParamTrialFESpace(test,g)
end

function MyTrial(
  test::SingleFieldFESpace,
  g::Function,
  ::Val{false})
  ParamTransientTrialFESpace(test,g)
end

function MyTrial(
  test::SingleFieldFESpace,
  g::Function,
  ptype::ProblemType)
  MyTrial(test,g,issteady(ptype))
end

struct MyTrials{Ttr} <: MySpaces
  trial::Ttr
  trial_no_bc::SingleFieldFESpace
  ddofs_on_full_trian::Vector{Int}

  function MyTrials(
    trial::Ttr,
    trial_no_bc::SingleFieldFESpace) where Ttr

    ddofs_on_full_trian = dirichlet_dofs_on_full_trian(trial.space,trial_no_bc)
    new{Ttr}(trial,trial_no_bc,ddofs_on_full_trian)
  end
end

function MyTrials(tests::MyTests,args...)
  trial = MyTrial(tests.test,args...)
  trial_no_bc = TrialFESpace(tests.test_no_bc)
  MyTrials(trial,trial_no_bc)
end

function ParamAffineFEOperator(a::Function,b::Function,
  pspace::ParamSpace,trial::MyTrials,test::MyTests)
  ParamAffineFEOperator(a,b,pspace,get_trial(trial),get_test(test))
end

function ParamFEOperator(res::Function,jac::Function,
  pspace::ParamSpace,trial::MyTrials,test::MyTests)
  ParamFEOperator(res,jac,pspace,get_trial(trial),get_test(test))
end

function ParamMultiFieldFESpace(spaces::Vector{MyTrials})
  ParamMultiFieldFESpace([get_trial(first(spaces)),get_trial(last(spaces))])
end

function ParamMultiFieldFESpace(spaces::Vector{MyTests})
  ParamMultiFieldFESpace([get_test(first(spaces)),get_test(last(spaces))])
end

function ParamTransientAffineFEOperator(m::Function,a::Function,b::Function,
  pspace::ParamSpace,trial::MyTrials,test::MyTests)
  ParamTransientAffineFEOperator(m,a,b,pspace,get_trial(trial),get_test(test))
end

function ParamTransientFEOperator(res::Function,jac::Function,jac_t::Function,
  pspace::ParamSpace,trial::MyTrials,test::MyTests)
  ParamTransientFEOperator(res,jac,jac_t,pspace,get_trial(trial),get_test(test))
end

function ParamTransientMultiFieldFESpace(spaces::Vector{MyTrials})
  ParamTransientMultiFieldFESpace([get_trial(first(spaces)),get_trial(last(spaces))])
end

function ParamTransientMultiFieldFESpace(spaces::Vector{MyTests})
  ParamTransientMultiFieldFESpace([get_test(first(spaces)),get_test(last(spaces))])
end

function free_dofs_on_full_trian(tests::MyTests)
  nfree_on_full_trian = tests.test_no_bc.nfree
  setdiff(collect(1:nfree_on_full_trian),tests.ddofs_on_full_trian)
end

function free_dofs_on_full_trian(trials::MyTrials)
  nfree_on_full_trian = trials.trial_no_bc.nfree
  setdiff(collect(1:nfree_on_full_trian),trials.ddofs_on_full_trian)
end

function get_fd_dofs(tests::MyTests,trials::MyTrials)
  fdofs_test = free_dofs_on_full_trian(tests)
  fdofs_trial = free_dofs_on_full_trian(trials)
  ddofs = trials.ddofs_on_full_trian
  (fdofs_test,fdofs_trial),ddofs
end

function Gridap.get_background_model(test::SingleFieldFESpace)
  get_background_model(get_triangulation(test))
end

function get_dimension(test::SingleFieldFESpace)
  model = get_background_model(test)
  maximum(model.grid.reffes[1].reffe.polytope.dface.dims)
end

function Gridap.FESpaces.get_order(test::SingleFieldFESpace)
  basis = get_fe_basis(test)
  first(basis.cell_basis.values[1].fields.orders)
end

Gridap.FESpaces.get_test(tests::MyTests) = tests.test
Gridap.FESpaces.get_trial(trials::MyTrials) = trials.trial
get_test_no_bc(tests::MyTests) = tests.test_no_bc
get_trial_no_bc(trials::MyTrials) = trials.trial_no_bc
get_degree(order::Int,c=2) = c*order
get_degree(test::SingleFieldFESpace,c=2) = get_degree(Gridap.FESpaces.get_order(test),c)
realization(fes::MySpaces) = FEFunction(fes.test,rand(num_free_dofs(fes.test)))

function get_cell_quadrature(test::SingleFieldFESpace)
  CellQuadrature(get_triangulation(test),get_degree(test))
end

struct LagrangianQuadFESpace
  test::SingleFieldFESpace

  function LagrangianQuadFESpace(model::DiscreteModel,order::Int)
    reffe_quad = Gridap.ReferenceFE(lagrangian_quad,Float,order)
    test = TestFESpace(model,reffe_quad,conformity=:L2)
    new(test)
  end
end

function LagrangianQuadFESpace(test::SingleFieldFESpace)
  model = get_background_model(test)
  order = Gridap.FESpaces.get_order(test)
  LagrangianQuadFESpace(model,order)
end

function LagrangianQuadFESpace(tests::MyTests)
  LagrangianQuadFESpace(get_test(tests))
end

function Gridap.FEFunction(
  quad_fespace::LagrangianQuadFESpace,
  vec::AbstractVector)

  FEFunction(quad_fespace.test,vec)
end

function Gridap.FEFunction(
  quad_fespace::LagrangianQuadFESpace,
  mat::AbstractMatrix)

  n -> FEFunction(quad_fespace.test,mat[:,n])
end

function get_phys_quad_points(test::SingleFieldFESpace)
  trian = get_triangulation(test)
  phys_map = get_cell_map(trian)
  cell_quad = get_cell_quadrature(test)
  cell_points = get_data(get_cell_points(cell_quad))
  map(Gridap.evaluate,phys_map,cell_points)
end
