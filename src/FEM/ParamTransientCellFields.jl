struct PTFunction <: Function
  f::Function
  params::AbstractArray
  times::Union{Real,Vector{<:Real}}
end

function get_fields(ptf::PTFunction)
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

# # FIELDS

# const PTField = Union{GenericField{PTFunction},
#                       FieldGradient{N,GenericField{PTFunction}} where N}
# const GenericPTField = Union{PTField,ZeroField{PTField}}

# function get_params(a::GenericField{PTFunction})
#   a.object.params
# end

# function get_times(a::GenericField{PTFunction})
#   a.object.times
# end

# function get_field(a::GenericField{PTFunction},p,t)
#   GenericField(a.object.f(p,t))
# end

# function get_params(a::FieldGradient{N,GenericField{PTFunction}} where N)
#   get_params(a.object)
# end
# function get_times(a::FieldGradient{N,GenericField{PTFunction}} where N)
#   get_times(a.object)
# end

# function get_field(a::FieldGradient{N,GenericField{PTFunction}} where N,p,t)
#   FieldGradient{N}(get_field(a.object,p,t))
# end

# function get_nfields(a::PTField)
#   length(get_params(a))*length(get_times(a))
# end

# function Arrays.testitem(a::PTField)
#   p,t = get_params(a),get_times(a)
#   p1,t1 = map(testitem,(p,t))
#   f = get_field(a,p1,t1)
#   return f
# end

# function Arrays.return_cache(fpt::GenericPTField,x::AbstractArray{<:Point})
#   n = get_nfields(fpt)
#   f = testitem(fpt)
#   cb,cf = return_cache(f,x)
#   ca = PTArray(fill(cb.array,n))
#   ca,cb,cf
# end

# function Arrays.evaluate!(cache,fpt::GenericPTField,x::AbstractArray{<:Point})
#   ca,c... = cache
#   p,t = get_params(fpt),get_times(fpt)
#   np = length(p)
#   @inbounds for q = eachindex(ca)
#     pq = p[fast_idx(q,np)]
#     tq = t[slow_idx(q,np)]
#     fq = get_field(fpt,pq,tq)
#     ca[q] = evaluate!(c,fq,x)
#   end
#   ca
# end

# CELLFIELDS
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

CellData.get_data(f::GenericPTCellField) = f.cell_field
CellData.get_triangulation(f::GenericPTCellField) = f.trian
CellData.DomainStyle(::Type{GenericPTCellField{DS}}) where DS = DS()
function CellData.similar_cell_field(::PTCellField,cell_data,trian,ds)
  GenericPTCellField(cell_data,trian,ds)
end

function CellData.CellField(f::PTFunction,trian::Triangulation,::DomainStyle)
  s = size(get_cell_map(trian))
  cell_field = PTArray(map(x->Fill(x,s),get_fields(f)))
  GenericPTCellField(cell_field,trian,PhysicalDomain())
end

function CellData.CellField(fs::SingleFieldFESpace,cell_vals::PTArray)
  v = get_fe_basis(fs)
  cell_basis = get_data(v)
  cell_field = PTArray(map(x->lazy_map(linear_combination,x,cell_basis),cell_vals.array))
  GenericPTCellField(cell_field,get_triangulation(v),DomainStyle(v))
end

function CellData.change_domain_ref_ref(
  a::PTCellField,ttrian::Triangulation,sglue::FaceToFaceGlue,tglue::FaceToFaceGlue)
  sface_to_fields = get_data(a)
  mface_to_sface = sglue.mface_to_tface
  tface_to_mface = tglue.tface_to_mface
  tface_to_mface_map = tglue.tface_to_mface_map
  ptarray = map(sface_to_fields) do sface_to_field
    mface_to_field = extend(sface_to_field,mface_to_sface)
    tface_to_field_s = lazy_map(Reindex(mface_to_field),tface_to_mface)
    tface_to_field_t = lazy_map(Broadcasting(∘),tface_to_field_s,tface_to_mface_map)
    tface_to_field_t
  end
  tface_to_fields_t = PTArray(ptarray)
  similar_cell_field(a,tface_to_fields_t,ttrian,ReferenceDomain())
end

function CellData.change_domain_phys_phys(
  a::PTCellField,ttrian::Triangulation,sglue::FaceToFaceGlue,tglue::FaceToFaceGlue)
  sface_to_fields = get_data(a)
  mface_to_sface = sglue.mface_to_tface
  tface_to_mface = tglue.tface_to_mface
  ptarray = map(sface_to_fields) do sface_to_field
    mface_to_field = extend(sface_to_field,mface_to_sface)
    tface_to_field = lazy_map(Reindex(mface_to_field),tface_to_mface)
    tface_to_field
  end
  tface_to_fields = PTArray(ptarray)
  similar_cell_field(a,tface_to_fields,ttrian,PhysicalDomain())
end

abstract type PTFEFunction <: PTCellField end

struct PTSingleFieldFEFunction{T<:CellField} <: PTFEFunction
  cell_field::T
  cell_dof_values::PTArray{<:AbstractArray{<:AbstractVector{<:Number}}}
  free_values::PTArray{<:AbstractVector{<:Number}}
  dirichlet_values::PTArray{<:AbstractVector{<:Number}}
  fe_space::SingleFieldFESpace
end

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

for T in (:ConstantFESpace,:DirichletFESpace,:FESpaceWithConstantFixed,
  :FESpaceWithLinearConstraints,:UnconstrainedFESpace)
  @eval begin
    function FESpaces.scatter_free_and_dirichlet_values(
      f::$T,
      free_values::PTArray,
      dirichlet_values::PTArray)

      ptarrays = map((fv,dv) -> scatter_free_and_dirichlet_values(f,fv,dv),
        free_values.array,dirichlet_values.array)
      PTArray(ptarrays)
    end
  end
end

const PTSingleFieldTypes = Union{GenericCellField,PTSingleFieldFEFunction}

function TransientCellField(single_field::PTSingleFieldTypes,derivatives::Tuple)
  TransientSingleFieldCellField(single_field,derivatives)
end

struct PTMultiFieldFEFunction{T<:MultiFieldCellField} <: PTFEFunction
  single_fe_functions::Vector{<:PTSingleFieldFEFunction}
  free_values::AbstractArray
  fe_space::MultiFieldFESpace
  multi_cell_field::T

  function PTMultiFieldFEFunction(
    free_values::AbstractVector,
    space::MultiFieldFESpace,
    single_fe_functions::Vector{<:PTSingleFieldFEFunction})

    multi_cell_field = MultiFieldCellField(map(i->i.cell_field,single_fe_functions))
    T = typeof(multi_cell_field)

    new{T}(
      single_fe_functions,
      free_values,
      space,
      multi_cell_field)
  end
end

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
  blockmask = [ is_change_possible(get_triangulation(uh),trian) for uh in uhs ]
  active_block_ids = findall(blockmask)
  active_block_data = Any[ get_cell_dof_values(uhs[i],trian) for i in active_block_ids ]
  nblocks = length(uhs)
  lazy_map(BlockMap(nblocks,active_block_ids),active_block_data...)
end

num_fields(m::PTMultiFieldFEFunction) = length(m.single_fe_functions)
Base.iterate(m::PTMultiFieldFEFunction) = iterate(m.single_fe_functions)
Base.iterate(m::PTMultiFieldFEFunction,state) = iterate(m.single_fe_functions,state)
Base.getindex(m::PTMultiFieldFEFunction,field_id::Integer) = m.single_fe_functions[field_id]

const PTMultiFieldTypes = Union{MultiFieldCellField,PTMultiFieldFEFunction}

function TransientCellField(multi_field::MultiFieldTypes,derivatives::Tuple)
  transient_single_fields = _to_transient_single_fields(multi_field,derivatives)
  TransientMultiFieldCellField(multi_field,derivatives,transient_single_fields)
end

# op,solver = feop,fesolver
# a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
# a(μ,t) = x->a(x,μ,t)
# v = get_fe_basis(test)
# u = get_trial_fe_basis(allocate_trial_space(trial))
# params = realization(op,2)
# times = get_times(solver)
# a(params[1],times[1])*∇(v)⋅∇(u)
# pf = PTFunction(a,params,times)
# F = pf*∇(v)⋅∇(u)
# quad = dΩ.quad
# # ∫(pf*∇(v)⋅∇(u))dΩ
# trian_f = get_triangulation(F)
# trian_x = get_triangulation(dΩ.quad)

# b = change_domain(F,quad.trian,quad.data_domain_style)
# x = get_cell_points(quad)
# bx = b(x)
# cell_map = get_cell_map(quad.trian)
# cell_Jt = lazy_map(∇,cell_map)
# cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
# lazy_map(IntegrationMap(),bx,quad.cell_weight,cell_Jtx)

# μ,t = realization(op),dt
# z(x,μ,t) = x[1]*μ[1]*t
# z(μ,t) = x->z(x,μ,t)
# zft(μ,t) = PTFunction(z,μ,t)

# myform(μ,t) = ∫(zft(μ,t)*∇(v)⋅∇(u))dΩ
# myform(params,times)
