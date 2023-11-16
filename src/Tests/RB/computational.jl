μ = Table([rand(3) for _ = 1:10])
t = rand(10)
cell_to_parent_cell = rand(1:num_free_dofs(test),10)

dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(nothing,nothing))
trian = view(Ω,cell_to_parent_cell)
meas = Measure(trian,2)
@time ∫(aμt(μ,t)*∇(dv)⋅∇(du))meas

rmodel = Geometry.DiscreteModelPortion(model,cell_to_parent_cell)
rtrian = TriangulationWithTags(rmodel)
rmeas = Measure(rtrian,2)
rtest = TestFESpace(rmodel,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
rtrial = PTTrialFESpace(rtest,g)
rdv = get_fe_basis(rtest)
rdu = get_trial_fe_basis(rtrial(nothing,nothing))
@time ∫(aμt(μ,t)*∇(rdv)⋅∇(rdu))rmeas

red_feop = reduce_fe_operator(feop,rmodel)
myrtest = red_feop.test
myrtest.ndirichlet

vector_type = nothing
constraint = nothing
glue,Keys = test.metadata
@unpack cell_reffe,conformity,labels,dirichlet_tags,dirichlet_masks = Keys
# FESpace(rmodel,cell_reffe;conformity,labels,dirichlet_tags,dirichlet_masks)
conf = Conformity(testitem(cell_reffe),conformity)
Keys = FEKeys(cell_reffe,conformity,labels,dirichlet_tags,dirichlet_masks)
@assert FESpaces._use_clagrangian(Triangulation(rmodel),cell_reffe,conf) && num_vertices(rmodel) == num_nodes(rmodel)
# FESpaces._unsafe_clagrangian(cell_reffe,Triangulation(model),labels,
# vector_type,dirichlet_tags,dirichlet_masks,Triangulation(rmodel),keys)
ctype_reffe,cell_ctype = compress_cell_data(cell_reffe)
prebasis = get_prebasis(first(ctype_reffe))
T = return_type(prebasis)
node_to_tag = get_face_tag_index(labels,dirichlet_tags,0)
_vector_type = isnothing(vector_type) ? Vector{Float64} : vector_type
tag_to_mask = isnothing(dirichlet_masks) ? fill(FESpaces._default_mask(T),length(dirichlet_tags)) : dirichlet_masks
