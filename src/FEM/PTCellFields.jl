abstract type AbstractPTFunction{P,T} <: Function end

struct PFunction{P} <: AbstractPTFunction{P,nothing}
  f::Function
  params::P

  function PFunction(f::Function,params::P) where P
    new{P}(f,params)
  end
end

function GenericPTField(pf::PFunction{<:AbstractVector{<:Number}})
  p = pf.params
  GenericField(pf.f(p))
end

function GenericPTField(pf::PFunction)
  p = pf.params
  np = length(p)
  fields = Vector{GenericField}(undef,np)
  @inbounds for k = eachindex(p)
    pk = p[k]
    fields[k] = GenericField(pf.f(pk))
  end
  fields
end

struct PTFunction{P,T} <: AbstractPTFunction{P,T}
  f::Function
  params::P
  times::T

  function PTFunction(f::Function,params::P,times::T) where {P,T}
    new{P,T}(f,params,times)
  end
end

function GenericPTField(ptf::PTFunction{<:AbstractVector{<:Number},<:Real})
  p,t = ptf.params,ptf.times
  GenericField(ptf.f(p,t))
end

function GenericPTField(ptf::PTFunction{<:AbstractVector{<:Number},<:AbstractVector{<:Real}})
  p,t = ptf.params,ptf.times
  nt = length(t)
  fields = Vector{GenericField}(undef,nt)
  @inbounds for k = 1:nt
    tk = t[k]
    fields[k] = GenericField(ptf.f(p,tk))
  end
  fields
end

function GenericPTField(ptf::PTFunction)
  p,t = ptf.params,ptf.times
  np = length(p)
  nt = length(t)
  npt = np*nt
  fields = Vector{GenericField}(undef,npt)
  @inbounds for k = 1:npt
    pk = p[fast_idx(k,np)]
    tk = t[slow_idx(k,np)]
    fields[k] = GenericField(ptf.f(pk,tk))
  end
  fields
end

function evaluate!(cache,f::AbstractPTFunction,x::Point)
  g = GenericPTField(f)
  map(g) do gi
    gi(x)
  end
end

abstract type PTCellField <: CellField end

struct GenericPTCellField{DS} <: PTCellField
  cell_field::PTArray
  trian::Triangulation
  domain_style::DS
  function GenericPTCellField(
    cell_field::PTArray,
    trian::Triangulation,
    domain_style::DomainStyle)

    DS = typeof(domain_style)
    new{DS}(Fields.MemoArray(cell_field),trian,domain_style)
  end
end

Base.length(f::GenericPTCellField) = length(f.cell_field)
CellData.get_data(f::GenericPTCellField) = f.cell_field
CellData.get_triangulation(f::GenericPTCellField) = f.trian
CellData.DomainStyle(::Type{GenericPTCellField{DS}}) where DS = DS()

function CellData.similar_cell_field(::PTCellField,cell_data,trian,ds)
  GenericPTCellField(cell_data,trian,ds)
end

function CellData.CellField(
  f::AbstractPTFunction,
  trian::Triangulation,
  ::DomainStyle)

  s = size(get_cell_map(trian))
  cell_field = PTArray(lazy_map(x->Fill(x,s),GenericPTField(f)))
  GenericPTCellField(cell_field,trian,PhysicalDomain())
end

function CellData.CellField(
  f::AbstractPTFunction{<:AbstractVector{<:Number},T},
  trian::Triangulation,
  ::DomainStyle) where T

  s = size(get_cell_map(trian))
  cell_field = Fill(GenericPTField(f),s)
  GenericCellField(cell_field,trian,PhysicalDomain())
end

function CellData.CellField(fs::SingleFieldFESpace,cell_vals::PTArray)
  v = get_fe_basis(fs)
  cell_basis = get_data(v)
  cell_field = lazy_map(linear_combination,cell_vals,cell_basis)
  GenericPTCellField(cell_field,get_triangulation(v),DomainStyle(v))
end

function CellData.change_domain_ref_ref(
  a::PTCellField,ttrian::Triangulation,sglue::FaceToFaceGlue,tglue::FaceToFaceGlue)
  sface_to_fields = get_data(a)
  mface_to_sface = sglue.mface_to_tface
  tface_to_mface = tglue.tface_to_mface
  tface_to_mface_map = tglue.tface_to_mface_map
  tface_to_fields_t = map(sface_to_fields) do sface_to_field
    mface_to_field = extend(sface_to_field,mface_to_sface)
    tface_to_field_s = lazy_map(Reindex(mface_to_field),tface_to_mface)
    tface_to_field_t = lazy_map(Broadcasting(∘),tface_to_field_s,tface_to_mface_map)
    tface_to_field_t
  end
  similar_cell_field(a,tface_to_fields_t,ttrian,ReferenceDomain())
end

function CellData.change_domain_phys_phys(
  a::PTCellField,ttrian::Triangulation,sglue::FaceToFaceGlue,tglue::FaceToFaceGlue)
  sface_to_fields = get_data(a)
  mface_to_sface = sglue.mface_to_tface
  tface_to_mface = tglue.tface_to_mface
  tface_to_fields = map(sface_to_fields) do sface_to_field
    mface_to_field = extend(sface_to_field,mface_to_sface)
    tface_to_field = lazy_map(Reindex(mface_to_field),tface_to_mface)
    tface_to_field
  end
  similar_cell_field(a,tface_to_fields,ttrian,PhysicalDomain())
end

function Arrays.testitem(f::GenericPTCellField)
  GenericCellField(testitem(f.cell_field),f.trian,f.domain_style)
end

abstract type PTFEFunction <: PTCellField end

struct PTSingleFieldFEFunction{T<:CellField} <: PTFEFunction
  cell_field::T
  cell_dof_values::PTArray
  free_values::PTArray
  dirichlet_values::PTArray
  fe_space::SingleFieldFESpace
end

Base.length(f::PTSingleFieldFEFunction) = length(f.cell_field)
CellData.get_data(f::PTSingleFieldFEFunction) = get_data(f.cell_field)
CellData.get_triangulation(f::PTSingleFieldFEFunction) = get_triangulation(f.cell_field)
CellData.DomainStyle(::Type{PTSingleFieldFEFunction{T}}) where T = DomainStyle(T)

FESpaces.get_free_dof_values(f::PTSingleFieldFEFunction) = f.free_values
FESpaces.get_cell_dof_values(f::PTSingleFieldFEFunction) = f.cell_dof_values
FESpaces.get_fe_space(f::PTSingleFieldFEFunction) = f.fe_space

function FESpaces.FEFunction(
  fs::SingleFieldFESpace,
  free_values::PTArray,
  dirichlet_values::PTArray)

  cell_vals = scatter_free_and_dirichlet_values(fs,free_values,dirichlet_values)
  cell_field = CellField(fs,cell_vals)
  PTSingleFieldFEFunction(cell_field,cell_vals,free_values,dirichlet_values,fs)
end

function FESpaces.FEFunction(
  fs::SingleFieldFESpace,
  free_values::PTArray,
  dirichlet_values::AbstractArray)

  dv = PTArray(dirichlet_values,length(free_values))
  FEFunction(fs,free_values,dv)
end

function Arrays.testitem(f::PTSingleFieldFEFunction)
  cell_field = testitem(f.cell_field)
  cell_dof_values = testitem(f.cell_dof_values)
  free_values = testitem(f.free_values)
  dirichlet_values = testitem(f.dirichlet_values)
  fe_space = testitem(f.fe_space)
  SingleFieldFEFunction(cell_field,cell_dof_values,free_values,dirichlet_values,fe_space)
end

const PTSingleFieldTypes = Union{GenericCellField,PTSingleFieldFEFunction}

function TransientCellField(single_field::PTSingleFieldTypes,derivatives::Tuple)
  TransientSingleFieldCellField(single_field,derivatives)
end

struct PTMultiFieldFEFunction{T<:MultiFieldCellField} <: PTFEFunction
  single_fe_functions::Vector{<:PTSingleFieldFEFunction}
  free_values::Vector{<:PTArray}
  fe_space::MultiFieldFESpace
  multi_cell_field::T

  function PTMultiFieldFEFunction(
    free_values::Vector{<:PTArray},
    space::MultiFieldFESpace,
    single_fe_functions::Vector{<:PTSingleFieldFEFunction})

    multi_cell_field = MultiFieldCellField(map(i->i.cell_field,single_fe_functions))
    T = typeof(multi_cell_field)
    new{T}(single_fe_functions,free_values,space,multi_cell_field)
  end
end

Base.length(f::PTMultiFieldFEFunction) = length(first(f.single_fe_functions))
CellData.get_data(f::PTMultiFieldFEFunction) = get_data(f.multi_cell_field)
CellData.get_triangulation(f::PTMultiFieldFEFunction) = get_triangulation(f.multi_cell_field)
CellData.DomainStyle(::Type{PTMultiFieldFEFunction{T}}) where T = DomainStyle(T)
FESpaces.get_free_dof_values(f::PTMultiFieldFEFunction) = f.free_values
FESpaces.get_fe_space(f::PTMultiFieldFEFunction) = f.fe_space

function FESpaces.get_cell_dof_values(f::PTMultiFieldFEFunction)
  msg = """\n
  This method does not make sense for multi-field
  since each field can be defined on a different triangulation.
  Pass a triangulation in the second argument to get the DOF values
  on top of the corresponding cells.
  """
  trians = map(get_triangulation,f.fe_space.spaces)
  trian = first(trians)
  @check all(map(t->is_change_possible(t,trian),trians)) msg
  get_cell_dof_values(f,trian)
end

function FESpaces.get_cell_dof_values(f::PTMultiFieldFEFunction,trian::Triangulation)
  uhs = f.single_fe_functions
  blockmask = [is_change_possible(get_triangulation(uh),trian) for uh in uhs]
  active_block_ids = findall(blockmask)
  active_block_data = Any[ get_cell_dof_values(uhs[i],trian) for i in active_block_ids ]
  nblocks = length(uhs)
  lazy_map(BlockMap(nblocks,active_block_ids),active_block_data...)
end

MultiField.num_fields(m::PTMultiFieldFEFunction) = length(m.single_fe_functions)
Base.iterate(m::PTMultiFieldFEFunction) = iterate(m.single_fe_functions)
Base.iterate(m::PTMultiFieldFEFunction,state) = iterate(m.single_fe_functions,state)
Base.getindex(m::PTMultiFieldFEFunction,field_id::Integer) = m.single_fe_functions[field_id]

const PTMultiFieldTypes = Union{MultiFieldCellField,PTMultiFieldFEFunction}

function TransientCellField(multi_field::PTMultiFieldTypes,derivatives::Tuple)
  transient_single_fields = _to_transient_single_fields(multi_field,derivatives)
  TransientMultiFieldCellField(multi_field,derivatives,transient_single_fields)
end

function FESpaces.EvaluationFunction(fe::MultiFieldFESpace,free_values::PTArray)
  blocks = map(1:length(fe.spaces)) do i
    free_values_i = restrict_to_field(fe,free_values,i)
    fe_function_i = EvaluationFunction(fe.spaces[i],free_values_i)
    free_values_i,fe_function_i
  end
  free_values = first.(blocks)
  fe_functions = last.(blocks)
  PTMultiFieldFEFunction(free_values,fe,fe_functions)
end

function MultiField.restrict_to_field(
  f::MultiFieldFESpace,
  free_values::PTArray,
  field::Integer)

  MultiField._restrict_to_field(f,MultiFieldStyle(f),free_values,field)
end

function MultiField._restrict_to_field(
  f::MultiFieldFESpace,
  ::ConsecutiveMultiFieldStyle,
  free_values::PTArray,
  field::Integer)

  offsets = compute_field_offsets(f)
  U = f.spaces
  pini = offsets[field] + 1
  pend = offsets[field] + num_free_dofs(U[field])
  map(fv -> SubVector(fv,pini,pend),free_values)
end

function Arrays.testitem(f::PTMultiFieldFEFunction)
  single_fe_functions = map(testitem,f.single_fe_functions)
  free_values = map(testitem,f.free_values)
  fe_space = testitem(f.fe_space)
  MultiFieldFEFunction(free_values,fe_space,single_fe_functions)
end
