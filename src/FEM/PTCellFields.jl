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
FESpaces.get_triangulation(f::GenericPTCellField) = f.trian
CellData.DomainStyle(::Type{GenericPTCellField{DS}}) where DS = DS()

function CellData.similar_cell_field(::PTCellField,cell_data,trian,ds)
  GenericPTCellField(cell_data,trian,ds)
end

function CellData.CellField(
  f::AbstractPTFunction,
  trian::Triangulation,
  ::DomainStyle)

  s = size(get_cell_map(trian))
  x = get_cell_points(trian) |> get_data
  ptf = get_fields(f)
  ptcell_field = NonaffinePTArray(map(x->Fill(x,s),ptf))
  GenericPTCellField(ptcell_field,trian,PhysicalDomain())
end

function CellData.CellField(
  f::AbstractPTFunction{<:AbstractVector{<:Number},<:Union{Real,Nothing}},
  trian::Triangulation,
  ::DomainStyle)

  s = size(get_cell_map(trian))
  cell_field = Fill(get_fields(f),s)
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

struct PTOperationCellField{DS} <: PTCellField
  op::Operation
  args::Tuple
  trian::Triangulation
  domain_style::DS
  memo::Dict{Any,Any}

  function PTOperationCellField(op::Operation,args::CellField...;initial_check=false)
    @assert length(args) > 0
    trian = get_triangulation(first(args))
    domain_style = DomainStyle(first(args))
    @check all( map(i->DomainStyle(i)==domain_style,args) )
    if initial_check && num_cells(trian)>0
      x = _get_cell_points(args...)
      ax = map(i->i(x),args)
      axi = map(first,ax)
      Fields.BroadcastingFieldOpMap(op.op)(axi...)
    end
    new{typeof(domain_style)}(op,args,trian,domain_style,Dict())
  end
end

function CellData._get_cell_points(args::CellField...)
  k = findfirst(i->isa(i,CellState),args)
  if k === nothing
    j = findall(i->isa(i,Union{OperationCellField,PTOperationCellField}),args)
    if length(j) == 0
      CellData._get_cell_points(first(args))
    else
      CellData._get_cell_points(args[j]...)
    end
  else
    args[k].points
  end
end

function CellData._get_cell_points(a::PTOperationCellField...)
  b = []
  for ai in a
    for i in ai.args
      push!(b,i)
    end
  end
  CellData._get_cell_points(b...)
end

function CellData._get_cell_points(a::PTOperationCellField)
  CellData._get_cell_points(a.args...)
end

function CellData.get_data(f::PTOperationCellField)
  a = map(get_data,f.args)
  lazy_map(Broadcasting(f.op),a...)
end

FESpaces.get_triangulation(f::PTOperationCellField) = f.trian
CellData.DomainStyle(::Type{PTOperationCellField{DS}}) where DS = DS()

function Arrays.evaluate!(cache,f::PTOperationCellField,x::CellPoint)
  ax = map(i->i(x),f.args)
  lazy_map(Fields.BroadcastingFieldOpMap(f.op.op),ax...)
end

function CellData.change_domain(
  f::PTOperationCellField,
  target_trian::Triangulation,
  target_domain::DomainStyle)

  args = map(i->change_domain(i,target_trian,target_domain),f.args)
  PTOperationCellField(f.op,args...)
end

function CellData._operate_cellfields(k::Operation,a...)
  b = _to_common_domain(a...)
  if any(x->isa(x,Union{PTFEFunction,PTCellField,PTArray}),b)
    PTOperationCellField(k,b...)
  else
    OperationCellField(k,b...)
  end
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
FESpaces.get_triangulation(f::PTSingleFieldFEFunction) = get_triangulation(f.cell_field)
CellData.DomainStyle(::Type{PTSingleFieldFEFunction{T}}) where T = DomainStyle(T)

FESpaces.get_free_dof_values(f::PTSingleFieldFEFunction) = f.free_values
FESpaces.get_cell_dof_values(f::PTSingleFieldFEFunction) = f.cell_dof_values
FESpaces.get_fe_space(f::PTSingleFieldFEFunction) = f.fe_space

function FEFunction(
  fs::SingleFieldFESpace,
  free_values::PTArray,
  dirichlet_values::PTArray)

  cell_vals = scatter_free_and_dirichlet_values(fs,free_values,dirichlet_values)
  cell_field = CellField(fs,cell_vals)
  PTSingleFieldFEFunction(cell_field,cell_vals,free_values,dirichlet_values,fs)
end

function FEFunction(
  fs::SingleFieldFESpace,
  free_values::PTArray,
  dirichlet_values::AbstractArray)

  dv = AffinePTArray(dirichlet_values,length(free_values))
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

abstract type PTTransientCellField <: PTCellField end

CellData.get_data(f::PTTransientCellField) = @abstractmethod
FESpaces.get_triangulation(f::PTTransientCellField) = @abstractmethod
CellData.DomainStyle(::Type{PTTransientCellField}) =  @abstractmethod
Fields.gradient(f::PTTransientCellField) =  @abstractmethod
Fields.∇∇(f::PTTransientCellField) =  @abstractmethod
function CellData.change_domain(f::PTTransientCellField,trian::Triangulation,::DomainStyle)
  @abstractmethod
end

struct PTTransientSingleFieldCellField{A} <: PTTransientCellField
  cellfield::A
  derivatives::Tuple
end

const PTSingleFieldTypes = Union{GenericPTCellField,PTSingleFieldFEFunction}

function TransientFETools.TransientCellField(single_field::PTSingleFieldTypes,derivatives::Tuple)
  PTTransientSingleFieldCellField(single_field,derivatives)
end

CellData.get_data(f::PTTransientSingleFieldCellField) = get_data(f.cellfield)
FESpaces.get_triangulation(f::PTTransientSingleFieldCellField) = get_triangulation(f.cellfield)
CellData.DomainStyle(::Type{<:PTTransientSingleFieldCellField{A}}) where A = DomainStyle(A)
Fields.gradient(f::PTTransientSingleFieldCellField) = gradient(f.cellfield)
Fields.∇∇(f::PTTransientSingleFieldCellField) = ∇∇(f.cellfield)
CellData.change_domain(f::PTTransientSingleFieldCellField,trian::Triangulation,target_domain::DomainStyle) = change_domain(f.cellfield,trian,target_domain)

# Skeleton related Operations
function Base.getproperty(f::PTTransientSingleFieldCellField, sym::Symbol)
  if sym in (:⁺,:plus,:⁻, :minus)
    derivatives = ()
    if sym in (:⁺,:plus)
      cellfield = CellFieldAt{:plus}(f.cellfield)
      for iderivative in f.derivatives
        derivatives = (derivatives...,CellFieldAt{:plus}(iderivative))
      end
    elseif sym in (:⁻, :minus)
      cellfield = CellFieldAt{:minus}(f.cellfield)
      for iderivative in f.derivatives
        derivatives = (derivatives...,CellFieldAt{:minus}(iderivative))
      end
    end
    return TransientSingleFieldCellField(cellfield,derivatives)
  else
    return getfield(f, sym)
  end
end

function TransientFETools.∂t(f::PTTransientCellField)
  cellfield,derivatives = Gridap.Helpers.first_and_tail(f.derivatives)
  TransientCellField(cellfield,derivatives)
end

TransientFETools.∂tt(f::PTTransientCellField) = ∂t(∂t(f))

struct PTMultiFieldCellField{DS<:DomainStyle} <: PTCellField
  single_fields::Vector{<:PTCellField}
  domain_style::DS
end

function PTMultiFieldCellField(single_fields::Vector{<:PTCellField})
  @assert length(single_fields) > 0
  f1 = first(single_fields)
  if any(map(i->DomainStyle(i)==ReferenceDomain(),single_fields))
    domain_style = ReferenceDomain()
  else
    domain_style = PhysicalDomain()
  end
  PTMultiFieldCellField{typeof(domain_style)}(single_fields,domain_style)
end

function CellData.get_data(f::PTMultiFieldCellField)
  s = """
  Function get_data is not implemented for PTMultiFieldCellField at this moment.
  You need to extract the individual fields and then evaluate them separatelly.

  If ever implement this, evaluating a `PTMultiFieldCellField` directly would provide,
  at each evaluation point, a tuple with the value of the different fields.
  """
  @notimplemented s
end

function FESpaces.get_triangulation(f::PTMultiFieldCellField)
  s1 = first(f.single_fields)
  trian = get_triangulation(s1)
  @check all(map(i->trian===get_triangulation(i),f.single_fields))
  trian
end
CellData.DomainStyle(::Type{PTMultiFieldCellField{DS}}) where DS = DS()
MultiField.num_fields(a::PTMultiFieldCellField) = length(a.single_fields)
Base.getindex(a::PTMultiFieldCellField,i::Integer) = a.single_fields[i]
Base.iterate(a::PTMultiFieldCellField)  = iterate(a.single_fields)
Base.iterate(a::PTMultiFieldCellField,state)  = iterate(a.single_fields,state)

struct PTMultiFieldFEFunction{T<:PTMultiFieldCellField} <: PTFEFunction
  single_fe_functions::Vector{<:PTSingleFieldFEFunction}
  free_values::PTArray
  fe_space::PMultiFieldFESpace
  multi_cell_field::T

  function PTMultiFieldFEFunction(
    free_values::PTArray,
    space::PMultiFieldFESpace,
    single_fe_functions::Vector{<:PTSingleFieldFEFunction})

    multi_cell_field = PTMultiFieldCellField(map(i->i.cell_field,single_fe_functions))
    T = typeof(multi_cell_field)
    new{T}(single_fe_functions,free_values,space,multi_cell_field)
  end
end

Base.length(f::PTMultiFieldFEFunction) = length(first(f.single_fe_functions))
CellData.get_data(f::PTMultiFieldFEFunction) = get_data(f.multi_cell_field)
FESpaces.get_triangulation(f::PTMultiFieldFEFunction) = get_triangulation(f.multi_cell_field)
CellData.DomainStyle(::Type{PTMultiFieldFEFunction{T}}) where T = DomainStyle(T)
FESpaces.get_free_dof_values(f::PTMultiFieldFEFunction) = f.free_values
FESpaces.get_fe_space(f::PTMultiFieldFEFunction) = f.fe_space
MultiField.num_fields(a::PTMultiFieldFEFunction) = length(a.single_fe_functions)

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

Base.iterate(m::PTMultiFieldFEFunction) = iterate(m.single_fe_functions)
Base.iterate(m::PTMultiFieldFEFunction,state) = iterate(m.single_fe_functions,state)
Base.getindex(m::PTMultiFieldFEFunction,::Colon) = m
Base.getindex(m::PTMultiFieldFEFunction,field_id::Integer) = m.single_fe_functions[field_id]

function Arrays.testitem(f::PTMultiFieldFEFunction)
  single_fe_functions = map(testitem,f.single_fe_functions)
  free_values = testitem(f.free_values)
  fe_space = testitem(f.fe_space)
  MultiFieldFEFunction(free_values,fe_space,single_fe_functions)
end

struct PTTransientMultiFieldCellField{A} <: PTTransientCellField
  cellfield::A
  derivatives::Tuple
  transient_single_fields::Vector{<:PTTransientCellField}
end

const PTMultiFieldTypes = Union{PTMultiFieldCellField,PTMultiFieldFEFunction}

function TransientFETools.TransientCellField(multi_field::PTMultiFieldTypes,derivatives::Tuple)
  transient_single_fields = _to_transient_single_fields(multi_field,derivatives)
  PTTransientMultiFieldCellField(multi_field,derivatives,transient_single_fields)
end

function CellData.get_data(::PTTransientMultiFieldCellField)
  s = """
  Function get_data is not implemented for PTTransientMultiFieldCellField at this moment.
  You need to extract the individual fields and then evaluate them separatelly.

  If ever implement this, evaluating a `PTMultiFieldCellField` directly would provide,
  at each evaluation point, a tuple with the value of the different fields.
  """
  @notimplemented s
end

FESpaces.get_triangulation(f::PTTransientMultiFieldCellField) = get_triangulation(f.cellfield)
CellData.DomainStyle(::Type{PTTransientMultiFieldCellField{A}}) where A = DomainStyle(A)
MultiField.num_fields(f::PTTransientMultiFieldCellField) = length(f.cellfield)
Fields.gradient(f::PTTransientMultiFieldCellField) = gradient(f.cellfield)
Fields.∇∇(f::PTTransientMultiFieldCellField) = ∇∇(f.cellfield)
CellData.change_domain(f::PTTransientMultiFieldCellField,trian::Triangulation,target_domain::DomainStyle) = change_domain(f.cellfield,trian,target_domain)

function Base.getindex(f::PTTransientMultiFieldCellField,ifield::Integer)
  single_field = f.cellfield[ifield]
  single_derivatives = ()
  for ifield_derivatives in f.derivatives
    single_derivatives = (single_derivatives...,getindex(ifield_derivatives,ifield))
  end
  PTTransientSingleFieldCellField(single_field,single_derivatives)
end

function Base.getindex(f::PTTransientMultiFieldCellField,indices::Vector{<:Int})
  cellfield = PTMultiFieldCellField(f.cellfield[indices],DomainStyle(f.cellfield))
  derivatives = ()
  for derivative in f.derivatives
    derivatives = (derivatives...,PTMultiFieldCellField(derivative[indices],DomainStyle(derivative)))
  end
  transient_single_fields = _to_transient_single_fields(cellfield,derivatives)
  PTTransientMultiFieldCellField(cellfield,derivatives,transient_single_fields)
end

function TransientFETools._to_transient_single_fields(
  multi_field::PTMultiFieldTypes,
  derivatives::Tuple)

  transient_single_fields = PTTransientCellField[]
  for ifield in 1:num_fields(multi_field)
    single_field = multi_field[ifield]
    single_derivatives = ()
    for ifield_derivatives in derivatives
      single_derivatives = (single_derivatives...,getindex(ifield_derivatives,ifield))
    end
    transient_single_field = PTTransientSingleFieldCellField(single_field,single_derivatives)
    push!(transient_single_fields,transient_single_field)
  end
  transient_single_fields
end

Base.iterate(f::PTTransientMultiFieldCellField)  = iterate(f.transient_single_fields)
Base.iterate(f::PTTransientMultiFieldCellField,state)  = iterate(f.transient_single_fields,state)

function TransientFETools.∂t(f::PTTransientMultiFieldCellField)
  cellfield, derivatives = Gridap.Helpers.first_and_tail(f.derivatives)
  transient_single_field_derivatives = TransientCellField[]
  for transient_single_field in f.transient_single_fields
    push!(transient_single_field_derivatives,∂t(transient_single_field))
  end
  PTTransientMultiFieldCellField(cellfield,derivatives,transient_single_field_derivatives)
end

TransientFETools.∂tt(f::PTTransientMultiFieldCellField) = ∂t(∂t(f))
