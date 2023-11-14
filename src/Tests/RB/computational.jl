μ = [rand(3) for _ = 1:10]
t = rand(10)
@time ∫(aμt(μ,t)*∇(dv)⋅∇(du))dΩ
@time ∫(aμt(μ,t)*∇(_dv)⋅∇(_du))_dΩ
@time ∫(aμt(μ,t)*∇(__dv)⋅∇(__du))__dΩ
@time ∫(a(μ[1],t[1])*∇(dv)⋅∇(du))dΩ

rtrian = view(Ω,idx)
rmeas = Measure(rtrian,2)
@time ∫(aμt(μ,t)*∇(dv)⋅∇(du))rmeas

###################
###################
###################
μ = Table([rand(3) for _ = 1:10])
t = rand(10)

cell_dof_ids = get_cell_dof_ids(feop.test,Ω)
node_to_parent_node = Int32.(rand(1:test.nfree,5))
cell_to_parent_cell = find_cells(node_to_parent_node,cell_dof_ids)
grid = get_grid(model)
red_grid = RBGridPortion(get_grid(model),cell_to_parent_cell)
red_model = RBDiscreteModelPortion(model,red_grid)
red_trian = Triangulation(red_model)
red_meas = Measure(red_trian,2)
red_test = reduce_test(get_test(feop),red_trian)

rmodel = Geometry.DiscreteModelPortion(model,cell_to_parent_cell)
rtest = TestFESpace(rmodel,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])

red_test.cell_dofs_ids == rtest.cell_dofs_ids
red_test.cell_is_dirichlet == rtest.cell_is_dirichlet
red_test.dirichlet_cells == rtest.dirichlet_cells
red_test.dirichlet_dof_tag == rtest.dirichlet_dof_tag
red_test.fe_basis == rtest.fe_basis
red_test.fe_dof_basis == rtest.fe_dof_basis
red_test.metadata == rtest.metadata
red_test.ndirichlet == rtest.ndirichlet
red_test.nfree == rtest.nfree
red_test.ntags == rtest.ntags
red_test.vector_type == rtest.vector_type
###################
###################
###################
μ = Table([rand(3) for _ = 1:10])
t = rand(10)

cell_dof_ids = get_cell_dof_ids(feop.test,Ω)
node_to_parent_node = Int32.(rand(1:test.nfree,5))
cell_to_parent_cell = find_cells(node_to_parent_node,cell_dof_ids)
grid = get_grid(model)
red_grid = RBGridPortion(get_grid(model),cell_to_parent_cell)
red_model = RBDiscreteModelPortion(model,red_grid)
red_trian = Triangulation(red_model)
red_meas = Measure(red_trian,2)
red_feop = reduce_feoperator(feop,red_trian)
dv_new = get_fe_basis(red_feop.test)
du_new = get_trial_fe_basis(red_feop.trials[1](nothing,nothing))
dc_new = ∫(aμt(μ,t)*∇(dv_new)⋅∇(du_new))red_meas
array_new = dc_new[red_trian]
x = NonaffinePTArray([zeros(red_feop.test.nfree) for _ = 1:100])
odeop_new = get_algebraic_operator(red_feop)
ode_cache_new = allocate_cache(odeop_new,μ,t)
op_new = get_ptoperator(odeop_new,μ,t,dt*θ,x,ode_cache_new,x)
Anew = allocate_jacobian(op_new,x)

rtrian = view(Ω,cell_to_parent_cell)
rmeas = Measure(rtrian,2)
dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(nothing,nothing))
dc = ∫(aμt(μ,t)*∇(dv)⋅∇(du))rmeas
array = dc[rtrian]
A = allocate_jacobian(op,x)

array_new ≈ array

#######
function reduced_node_coordinates(grid,cell_to_parent_cell)
  node_coordinates = collect1d(get_node_coordinates(grid))
  cell_node_ids = get_cell_node_ids(grid)[cell_to_parent_cell]
  node_ids = unique(sort(cell_node_ids.data))
  node_coordinates[node_ids]
end

trian = red_trian
model = get_background_model(trian)
cell_to_parent_cell = get_cell_to_parent_cell(model)
cell_dofs_ids = test.cell_dofs_ids[cell_to_parent_cell]
cell_is_dirichlet = test.cell_is_dirichlet[cell_to_parent_cell]
dirichlet_cells = intersect(test.dirichlet_cells,cell_to_parent_cell)
cell_basis = test.fe_basis.cell_basis[cell_to_parent_cell]
fe_basis = FESpaces.SingleFieldFEBasis(cell_basis,trian,FESpaces.TestBasis(),test.fe_basis.domain_style)
cell_dof_basis = test.fe_dof_basis.cell_dof[cell_to_parent_cell]
fe_dof_basis = CellDof(cell_dof_basis,trian,test.fe_dof_basis.domain_style)
dirichlet_dofs = get_reduced_dirichlet_dof_ids(test,dirichlet_cells)
free_dofs = get_reduced_free_dof_ids(cell_dofs_ids)
ndirichlet = length(dirichlet_dofs)
nfree = length(free_dofs)
ntags = test.ntags
vector_type = test.vector_type
dirichlet_dof_tag = test.dirichlet_dof_tag[dirichlet_dofs]
glue = test.metadata

dirichlet_dof_to_comp = glue.dirichlet_dof_to_comp[dirichlet_dofs]
dirichlet_dof_to_node = glue.dirichlet_dof_to_node[dirichlet_dofs]
free_dof_to_comp = glue.free_dof_to_comp[free_dofs]
free_dof_to_node = glue.free_dof_to_node[free_dofs]
# node_and_comp_to_dof = get_reduced_node_and_comp_to_dof(glue.node_and_comp_to_dof,free_dofs,dirichlet_dofs)
reduced_dofs = union(free_dofs,-dirichlet_dofs)
nreddofs = length(reduced_dofs)
reduced_node_and_comp_to_dof = Vector{Int32}(undef,nreddofs)
count = 1
for node_comp in glue.node_and_comp_to_dof
  if node_comp ∈ reduced_dofs
    reduced_node_and_comp_to_dof[count] = node_comp
    count += 1
  end
end
