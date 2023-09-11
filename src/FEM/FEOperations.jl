function collect_trian(a::DomainContribution)
  t = ()
  for trian in get_domains(a)
    t = (t...,trian)
  end
  unique(t)
end

function Base.:(==)(
  a::T,
  b::T
  ) where {T<:Union{Grid,Field}}

  for field in propertynames(a)
    a_field = getproperty(a,field)
    b_field = getproperty(b,field)
    if isa(a_field,GridapType)
      (==)(a_field,b_field)
    else
      if isdefined(a_field,1) && !(==)(a_field,b_field)
        return false
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

function modify_measures!(measures::Vector{Measure},m::Measure)
  for (nmeas,meas) in enumerate(measures)
    if is_parent(get_triangulation(meas),get_triangulation(m))
      measures[nmeas] = m
      return
    end
  end
  @unreachable "Unrecognizable measure"
end

function modify_measures(measures::Vector{Measure},m::Measure)
  new_measures = copy(measures)
  modify_measures!(new_measures,m)
  new_measures
end

Gridap.CellData.get_triangulation(m::Measure) = m.quad.trian

function FESpaces.get_order(test::SingleFieldFESpace)
  basis = get_fe_basis(test)
  first(FESpaces.get_order(first(basis.cell_basis.values).fields))
end

function FESpaces.get_order(test::MultiFieldFESpace)
  orders = map(get_order,test)
  maximum(orders)
end

Base.zeros(fe::FESpace) = get_free_dof_values(zero(fe))

# Remove when possible
function CellData.is_change_possible(
  strian::Triangulation,
  ttrian::Triangulation)

  msg = """\n
  Triangulations do not point to the same background discrete model!
  """

  if strian === ttrian
    return true
  end

  @check get_background_model(strian) == get_background_model(ttrian) msg

  D = num_cell_dims(strian)
  sglue = get_glue(strian,Val(D))
  tglue = get_glue(ttrian,Val(D))
  is_change_possible(sglue,tglue)
end

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

abstract type NormStyle end
struct l2Norm <: NormStyle end
struct L2Norm <: NormStyle end
struct H1Norm <: NormStyle end

get_norm_matrix(::l2Norm,args...) = nothing

get_norm_matrix(::L2Norm,args...) = get_L2_norm_matrix(args...)

get_norm_matrix(::H1Norm,args...) = get_H1_norm_matrix(args...)

for f in (:get_L2_norm_matrix,:get_H1_norm_matrix)
  @eval begin
    function $f(op::ParamTransientFEOperator)
      test = op.test
      trial = get_trial(op)
      trial_hom = allocate_trial_space(trial)
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
