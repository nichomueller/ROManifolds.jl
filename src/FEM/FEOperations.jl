function Base.:(==)(a::T,b::T) where {T<:Union{Grid,Field}}
  for field in propertynames(a)
    a_field = getproperty(a,field)
    b_field = getproperty(b,field)
    if isa(a_field,GridapType)
      (==)(a_field,b_field)
    else
      if isdefined(a_field,1) && !(==)(a_field,b_field)
        @assert false
      end
    end
  end
  return true
end

function is_parent(tparent::Triangulation,tchild::Triangulation)
  try
    try
      tparent.model == tchild.model && tparent.grid == tchild.grid.parent
    catch
      tparent == tchild.parent
    end
  catch
    false
  end
end

function FESpaces.is_change_possible(strian::Triangulation,ttrian::Triangulation)
  msg = "Triangulations do not point to the same background discrete model!"
  if strian == ttrian
    return true
  end
  @check get_background_model(strian) == get_background_model(ttrian) msg
  D = num_cell_dims(strian)
  sglue = get_glue(strian,Val(D))
  tglue = get_glue(ttrian,Val(D))
  is_change_possible(sglue,tglue)
end

function FESpaces.get_order(test::SingleFieldFESpace)
  basis = get_fe_basis(test)
  first(FESpaces.get_order(first(basis.cell_basis.values).fields))
end

function FESpaces.get_order(test::MultiFieldFESpace)
  orders = map(get_order,test)
  maximum(orders)
end

function field_offsets(f::MultiFieldFESpace)
  nfields = length(f.spaces)
  offsets = zeros(Int,nfields+1)
  @inbounds for field = 1:nfields
    offsets[field+1] = offsets[field] + num_free_dofs(f.spaces[field])
  end
  offsets
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
