struct FEKeys{T,M}
  cell_reffe::AbstractArray{<:ReferenceFE}
  conformity::Symbol
  dirichlet_tags::T
  dirichlet_masks::M
  function FEKeys(
    cell_reffe::AbstractArray{<:ReferenceFE},
    conformity::Symbol,
    dirichlet_tags::T,
    dirichlet_masks::M) where {T,M}
    new{T,M}(cell_reffe,conformity,dirichlet_tags,dirichlet_masks)
  end
end

function FESpaces._ConformingFESpace(
  vector_type::Type,
  model::DiscreteModel,
  face_labeling::FaceLabeling,
  cell_fe::CellFE,
  dirichlet_tags,
  dirichlet_components,
  trian=Triangulation(model),
  keys=nothing)

  grid_topology = get_grid_topology(model)
  ntags = length(dirichlet_tags)

  cell_dofs_ids,nfree,ndirichlet,dirichlet_dof_tag,dirichlet_cells = compute_conforming_cell_dofs(
    cell_fe,CellConformity(cell_fe),grid_topology,face_labeling,dirichlet_tags,dirichlet_components)

  cell_shapefuns, cell_dof_basis = compute_cell_space(cell_fe,trian)

  cell_is_dirichlet = fill(false,num_cells(trian))
  cell_is_dirichlet[dirichlet_cells] .= true

  UnconstrainedFESpace(vector_type,nfree,ndirichlet,cell_dofs_ids,cell_shapefuns,
    cell_dof_basis,cell_is_dirichlet,dirichlet_dof_tag,dirichlet_cells,ntags,keys)
end

function FESpaces._unsafe_clagrangian(
  cell_reffe,
  grid,
  labels,
  vector_type,
  dirichlet_tags,
  dirichlet_masks,
  trian=grid,
  keys=nothing)

  ctype_reffe,cell_ctype = compress_cell_data(cell_reffe)
  prebasis = get_prebasis(first(ctype_reffe))
  T = return_type(prebasis)
  node_to_tag = get_face_tag_index(labels,dirichlet_tags,0)
  _vector_type = isnothing(vector_type) ? Vector{Float64} : vector_type
  tag_to_mask = isnothing(dirichlet_masks) ? fill(FESpaces._default_mask(T),length(dirichlet_tags)) : dirichlet_masks
  CLagrangianFESpace(T,grid,_vector_type,node_to_tag,tag_to_mask,trian,keys)
end

function FESpaces.CLagrangianFESpace(
  ::Type{T},
  grid::Triangulation,
  vector_type::Type,
  node_to_tag::AbstractVector{<:Integer},
  tag_to_masks::AbstractVector,
  trian::Triangulation=grid,
  keys=nothing) where T

  z = zero(T)
  glue, dirichlet_dof_tag = FESpaces._generate_node_to_dof_glue_component_major(
    z,node_to_tag,tag_to_masks)
  cell_reffe = FESpaces._generate_cell_reffe_clagrangian(z,grid)
  cell_dofs_ids = FESpaces._generate_cell_dofs_clagrangian(z,grid,glue,cell_reffe)
  cell_shapefuns = lazy_map(get_shapefuns,cell_reffe)
  cell_dof_basis = lazy_map(get_dof_basis,cell_reffe)
  rd = ReferenceDomain()
  fe_basis, fe_dof_basis = compute_cell_space(
    cell_shapefuns,cell_dof_basis,rd,rd,trian)
  cell_is_dirichlet = FESpaces._generate_cell_is_dirichlet(cell_dofs_ids)

  nfree = length(glue.free_dof_to_node)
  ndirichlet = length(glue.dirichlet_dof_to_node)
  ntags = length(tag_to_masks)
  dirichlet_cells = collect(Int32,findall(cell_is_dirichlet))
  metadata = glue,keys

  UnconstrainedFESpace(vector_type,nfree,ndirichlet,cell_dofs_ids,fe_basis,
    fe_dof_basis,cell_is_dirichlet,dirichlet_dof_tag,dirichlet_cells,ntags,metadata)
end

function FESpaces.FESpace(
  model::DiscreteModel,
  cell_fe::CellFE;
  trian = Triangulation(model),
  labels = get_face_labeling(model),
  dirichlet_tags=Int[],
  dirichlet_masks=nothing,
  constraint=nothing,
  vector_type=nothing,
  keys=nothing)

  @assert num_cells(cell_fe) == num_cells(model) """\n
  The number of cells provided in the `cell_fe` argument ($(cell_fe.num_cells) cells)
  does not match the number of cells ($(num_cells(model)) cells) in the provided DiscreteModel.
  """
  _vector_type = FESpaces._get_vector_type(vector_type,cell_fe,trian)
  F = FESpaces._ConformingFESpace(
      _vector_type,
      model,
      labels,
      cell_fe,
      dirichlet_tags,
      dirichlet_masks,
      trian,
      keys)
  FESpaces._add_constraint(F,cell_fe.max_order,constraint)
end

function FESpaces.FESpace(
  model::DiscreteModel,
  cell_reffe::AbstractArray{<:ReferenceFE};
  conformity=nothing,
  trian=Triangulation(model),
  labels=get_face_labeling(model),
  dirichlet_tags=Int[],
  dirichlet_masks=nothing,
  constraint=nothing,
  vector_type=nothing,
  keep_keys=false)

  conf = Conformity(testitem(cell_reffe),conformity)

  keys = keep_keys ? FEKeys(cell_reffe,conformity,dirichlet_tags,dirichlet_masks) : nothing

  if FESpaces._use_clagrangian(trian,cell_reffe,conf) &&
    isnothing(constraint) &&
    num_vertices(model) == num_nodes(model)

    return FESpaces._unsafe_clagrangian(cell_reffe,Triangulation(model),labels,
      vector_type,dirichlet_tags,dirichlet_masks,trian,keys)
  end

  cell_fe = CellFE(model,cell_reffe,conf)
  _vector_type = FESpaces._get_vector_type(vector_type,cell_fe,trian)
  if conformity in (L2Conformity(),:L2) && dirichlet_tags == Int[]
    F = FESpaces._DiscontinuousFESpace(_vector_type,trian,cell_fe)
    V = FESpaces._add_constraint(F,cell_fe.max_order,constraint)
  else
    V = FESpace(model,cell_fe;
      trian=trian,
      labels=labels,
      dirichlet_tags=dirichlet_tags,
      dirichlet_masks=dirichlet_masks,
      constraint=constraint,
      vector_type=_vector_type,
      keys=keys)
  end
  return V
end

function reduce_fe_space(
  fs::UnconstrainedFESpace,
  model::DiscreteModelPortion)

  glue,keys = fs.metadata
  @unpack cell_reffe,conformity,dirichlet_tags,dirichlet_masks = keys
  FESpace(model,cell_reffe;conformity,dirichlet_tags,dirichlet_masks)
end

function reduce_fe_space(
  fs::UnconstrainedFESpace{V,FEKeys} where V,
  model::DiscreteModelPortion)

  keys = fs.metadata
  @unpack cell_reffe,conformity,dirichlet_tags,dirichlet_masks = keys
  FESpace(model,cell_reffe;conformity,dirichlet_tags,dirichlet_masks)
end

function reduce_fe_space(fs::FESpaceWithConstantFixed,model::DiscreteModelPortion)
  space = reduce_fe_space(fs.space,model)
  # dof_to_fix = num_free_dofs(space)
  # FESpaceWithConstantFixed(space,dof_to_fix)
  space
end

function reduce_fe_space(fs::ZeroMeanFESpace,model::DiscreteModelPortion)
  space = reduce_fe_space(fs.space,model)
  # FESpaceWithConstantFixed(space,fs.vol_i,fs.vol)
  space
end

function reduce_fe_operator(
  feop::PTFEOperatorFromWeakForm{Affine},
  model::DiscreteModelPortion)

  test = reduce_fe_space(get_test(feop),model)
  trial = PTTrialFESpace(test,get_trial(feop).dirichlet_μt)
  AffinePTFEOperator(feop.res,feop.jacs[1],feop.jacs[2],feop.pspace,trial,test)
end

function reduce_fe_operator(
  feop::PTFEOperatorFromWeakForm{Nonlinear},
  model::DiscreteModelPortion)

  test = reduce_fe_space(get_test(feop),model)
  trial = PTTrialFESpace(test,get_trial(feop).dirichlet_μt)
  PTFEOperator(feop.res,feop.jacs[1],feop.jacs[2],feop.nl,feop.pspace,trial,test)
end
