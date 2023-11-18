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

# ∇(rdu)
cell_∇a = lazy_map(Broadcasting(∇),get_data(rdu))
cell_map = get_cell_map(get_triangulation(rdu))
gfield = lazy_map(Broadcasting(push_∇),cell_∇a,cell_map)
# similar_cell_field(a,g,get_triangulation(a),DomainStyle(a))

# ∇(du)
cell_∇a = lazy_map(Broadcasting(∇),get_data(du))
cell_map = get_cell_map(get_triangulation(du))
gfield = lazy_map(Broadcasting(push_∇),cell_∇a,cell_map)
# similar_cell_field(a,g,get_triangulation(a),DomainStyle(a))

function temp_gradient(a::CellField,cell_to_parent_cell::AbstractVector)
  data = lazy_map(Reindex(get_data(a)),cell_to_parent_cell)
  cell_∇a = lazy_map(Broadcasting(∇),data)
  if DomainStyle(a) == PhysicalDomain()
    g = cell_∇a
  else
    cell_map = lazy_map(Reindex(get_cell_map(get_triangulation(a))),cell_to_parent_cell)
    g = lazy_map(Broadcasting(push_∇),cell_∇a,cell_map)
  end
  similar_cell_field(a,g,get_triangulation(a),DomainStyle(a))
end

@time ∫(aμt(μ,t)*temp_gradient(dv,cell_to_parent_cell)⋅temp_gradient(du,cell_to_parent_cell))meas

@time ∇(dv)⋅∇(du)
@time temp_gradient(dv,cell_to_parent_cell)⋅temp_gradient(du,cell_to_parent_cell)
@time ∇(rdv)⋅∇(rdu)

@time aμt(μ,t)*∇(dv)
@time aμt(μ,t)*temp_gradient(dv,cell_to_parent_cell)
@time aμt(μ,t)*∇(rdv)

@time aμt(μ,t)*∇(dv)⋅∇(du)
@time aμt(μ,t)*temp_gradient(dv,cell_to_parent_cell)⋅temp_gradient(du,cell_to_parent_cell)
@time aμt(μ,t)*∇(rdv)⋅∇(rdu)

@time ∫(aμt(μ,t)*∇(dv)⋅∇(du))meas
# @time aμt(μ,t)*temp_gradient(dv,cell_to_parent_cell)⋅temp_gradient(du,cell_to_parent_cell)
@time ∫(aμt(μ,t)*∇(rdv)⋅∇(rdu))rmeas

@time ∫(a(μ[1],t[1])*∇(dv)⋅∇(du))dΩ
@time ∫(a(μ[1],t[1])*∇(dv)⋅∇(du))meas
@time ∫(a(μ[1],t[1])*∇(rdv)⋅∇(rdu))rmeas

@time begin
  dc = ∫(a(μ[1],t[1])*∇(dv)⋅∇(du))dΩ
  matdata = collect_cell_matrix(trial(nothing,nothing),test,dc)
  assemble_matrix(feop.assem,matdata)
end

@time begin
  dc = ∫(a(μ[1],t[1])*∇(dv)⋅∇(du))meas
  matdata = collect_cell_matrix(trial(nothing,nothing),test,dc)
  assemble_matrix(feop.assem,matdata)
end

rassem = SparseMatrixAssembler(rtrial,rtest)
@time begin
  dc = ∫(a(μ[1],t[1])*∇(rdv)⋅∇(rdu))rmeas
  matdata = collect_cell_matrix(rtrial(nothing,nothing),rtest,dc)
  assemble_matrix(rassem,matdata)
end
