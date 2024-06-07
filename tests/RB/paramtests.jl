using Test
using Gridap.Arrays
using Gridap.TensorValues
using Gridap.ReferenceFEs
using Gridap.Geometry
using Gridap.Fields
using Gridap.Algebra
using SparseArrays
using SparseMatricesCSR
using Gridap.FESpaces
using Gridap.CellData
using Gridap.Algebra
using Mabla.FEM
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamAlgebra
using Mabla.FEM.ParamFESpaces

# using Test
# using Gridap.Arrays
# using Gridap.Algebra
# using Gridap.TensorValues
# using Gridap.ReferenceFEs
# using Gridap.Geometry
# using Gridap.Fields
# using Gridap.FESpaces
# using Gridap.CellData
# using Mabla.FEM

domain =(0,1,0,1,0,1)
partition = (3,3,3)
model = CartesianDiscreteModel(domain,partition)
trian = get_triangulation(model)
order = 2
degree = 4
quad = CellQuadrature(trian,degree)

μ = ParamRealization([[1],[2],[3]])
f(x,μ) = sum(μ)*x[1]
f(μ) = x -> f(x,μ)
fμ = 𝑓ₚ(f,μ)

l(x) = x[2]
liform(v) = v⊙l
biform(u,v) = fμ*∇(v)⊙∇(u)

reffe = ReferenceFE(lagrangian,Float64,order)
V = FESpace(model,reffe,dirichlet_tags=[1,10])
U = TrialParamFESpace(V,fμ)

v = get_fe_basis(V)
u = get_trial_fe_basis(U)

cellmat = integrate(biform(u,v),quad)
cellvec = integrate(liform(v),quad)
rows = get_cell_dof_ids(V,trian)
cols = get_cell_dof_ids(U,trian)
cellmat_c = attach_constraints_cols(U,cellmat,trian)
cellmat_rc = attach_constraints_rows(V,cellmat_c,trian)
cellvec_r = attach_constraints_rows(V,cellvec,trian)

assem = ParamFESpaces.get_param_assembler(SparseMatrixAssembler(U,V),μ)
matdata = ([cellmat_rc],[rows],[cols])
vecdata = ([cellvec_r],[rows])
A =  assemble_matrix(assem,matdata)
b =  assemble_vector(assem,vecdata)
x = A \ b
x0 = zero(x)

op = AffineFEOperator(U,V,AffineOperator(A,b))
solver = LinearFESolver()
test_fe_solver(solver,op,x0,x)
uh = solve(solver,op)
@test get_free_dof_values(uh) ≈ x
uh = solve(op)
@test get_free_dof_values(uh) ≈ x

# This is not supported for ParamArrays: have to use an algebraic solver
# solver = NonlinearFESolver()
# test_fe_solver(solver,op,x0,x)
# uh = solve(solver,op)
# @test get_free_dof_values(uh) ≈ x

nls = NewtonRaphsonSolver(LUSolver(),1e-10,20)
solver = NonlinearFESolver(nls)
test_fe_solver(solver,op,x0,x)

u0 = zero(U)
# b = allocate_residual(op,u0)
x = get_free_dof_values(u0)
# allocate_residual(op.op,x)
allocate_in_range(typeof(op.op.vector),op.op.matrix)


# Now using algebraic solvers directly
solver = LUSolver()
uh = solve(solver,op)
@test get_free_dof_values(uh) ≈ x
uh = solve(op)
@test get_free_dof_values(uh) ≈ x
uh,cache = solve!(uh,solver,op)
@test get_free_dof_values(uh) ≈ x
uh, = solve!(uh,solver,op,cache)

solver = NewtonRaphsonSolver(LUSolver(),1e-10,20)
uh = solve(solver,op)
@test get_free_dof_values(uh) ≈ x
uh = solve(op)
@test get_free_dof_values(uh) ≈ x
zh = zero(U)
zh,cache = solve!(zh,solver,op)
@test get_free_dof_values(zh) ≈ x
