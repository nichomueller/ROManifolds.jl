function is_parent(tparent::Triangulation,tchild::Triangulation;kwargs...)
  false
end

function is_parent(tparent::BodyFittedTriangulation,tchild::BodyFittedTriangulation;shallow=false)
  if shallow
    tparent.grid.cell_node_ids == tchild.grid.parent.cell_node_ids
  else
    tparent.grid === tchild.grid.parent
  end
end

function is_parent(tparent::BoundaryTriangulation,tchild::TriangulationView;shallow=false)
  if shallow
    (tparent.trian.grid.parent.cell_node_ids == tchild.parent.trian.grid.parent.cell_node_ids &&
      tparent.glue.face_to_cell == tchild.parent.glue.face_to_cell)
  else
    tparent === tchild.parent
  end
end

function correct_triangulation(trian::BodyFittedTriangulation,new_trian::BodyFittedTriangulation)
  old_grid = get_grid(trian)
  new_model = get_background_model(new_trian)
  new_grid = GridView(get_grid(new_trian),old_grid.cell_to_parent_cell)
  BodyFittedTriangulation(new_model,new_grid,trian.tface_to_mface)
end

function correct_triangulation(trian::TriangulationView,new_trian::BoundaryTriangulation)
  TriangulationView(new_trian,trian.cell_to_parent_cell)
end

function correct_quadrature(quad::CellQuadrature,new_trian::Triangulation)
  @unpack (cell_quad,cell_point,cell_weight,trian,
    data_domain_style,integration_domain_style) = quad
  CellQuadrature(cell_quad,cell_point,cell_weight,new_trian,data_domain_style,integration_domain_style)
end

function correct_measure(meas::Measure,trians::Triangulation...)
  trian = get_triangulation(meas)
  for t in trians
    if is_parent(t,trian;shallow=true)
      new_trian = correct_triangulation(trian,t)
      new_quad = correct_quadrature(meas.quad,new_trian)
      new_meas = Measure(new_quad)
      return new_meas
    end
  end
  @unreachable
end

function FESpaces.get_order(test::SingleFieldFESpace)
  basis = get_fe_basis(test)
  first(FESpaces.get_order(first(basis.cell_basis.values).fields))
end

function FESpaces.get_order(test::MultiFieldFESpace)
  orders = map(get_order,test)
  maximum(orders)
end

Base.zeros(fe::FESpace) = get_free_dof_values(zero(fe))

function get_discrete_model(
  tpath::String,
  mesh::String,
  bnd_info::Dict)

  mshpath = joinpath(get_parent_dir(tpath;nparent=3),"meshes/$mesh")
  if !ispath(mshpath)
    mshpath_msh_format = mshpath[1:findall(x->x=='.',mshpath)[end]-1]*".msh"
    model_msh_format = GmshDiscreteModel(mshpath_msh_format)
    to_json_file(model_msh_format,mshpath)
  end
  model = DiscreteModelFromFile(mshpath)
  set_labels!(model,bnd_info)
  model
end

function set_labels!(model,bnd_info)
  tags = collect(keys(bnd_info))
  bnds = collect(values(bnd_info))
  labels = get_face_labeling(model)
  for i = eachindex(tags)
    if tags[i] ∉ labels.tag_to_name
      add_tag_from_tags!(labels,tags[i],bnds[i])
    end
  end
end

for f in (:get_L2_norm_matrix,:get_H1_norm_matrix)
  @eval begin
    function $f(op::PTFEOperator)
      μ,t = realization(op),0.
      test = op.test
      trial = get_trial(op)
      trial_hom = allocate_trial_space(trial,μ,t)
      $f(test,trial_hom)
    end

    function $f(
      trial::TransientMultiFieldTrialFESpace,
      test::MultiFieldFESpace)

      map($f,trial.spaces,test.spaces)
    end
  end
end

function get_L2_norm_matrix(
  trial::TrialFESpace,
  test::FESpace)

  trian = get_triangulation(test)
  order = get_order(test)
  dΩ = Measure(trian,2*order)
  L2_form(u,v) = ∫(v⋅u)dΩ
  assemble_matrix(L2_form,trial,test)
end

function get_H1_norm_matrix(
  trial::TrialFESpace,
  test::FESpace)

  trian = get_triangulation(test)
  order = get_order(test)
  dΩ = Measure(trian,2*order)
  H1_form(u,v) = ∫(∇(v)⊙∇(u))dΩ + ∫(v⋅u)dΩ
  assemble_matrix(H1_form,trial,test)
end
