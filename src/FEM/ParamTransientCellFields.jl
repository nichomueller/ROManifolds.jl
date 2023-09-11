struct PTFunction <: Function
  f::Function
  params::AbstractArray
  times::Union{Real,Vector{<:Real}}
end

const PTField = Union{GenericField{PTFunction},
                      FieldGradient{N,GenericField{PTFunction}} where N}
const GenericPTField = Union{PTField,ZeroField{<:PTField}}

function get_params(a::GenericField{PTFunction})
  a.object.params
end

function get_times(a::GenericField{PTFunction})
  a.object.times
end

function get_field(a::GenericField{PTFunction},p,t)
  GenericField(a.object.f(p,t))
end

function get_params(a::FieldGradient{N,GenericField{PTFunction}} where N)
  get_params(a.object)
end
function get_times(a::FieldGradient{N,GenericField{PTFunction}} where N)
  get_times(a.object)
end

function get_field(a::FieldGradient{N,GenericField{PTFunction}} where N,p,t)
  FieldGradient{N}(get_field(a.object,p,t))
end

function get_nfields(a::PTField)
  length(get_params(a))*length(get_times(a))
end

function Arrays.testitem(a::PTField)
  p,t = get_params(a),get_times(a)
  p1,t1 = map(testitem,(p,t))
  f = get_field(a,p1,t1)
  return f
end

function Arrays.return_cache(fpt::GenericPTField,x::AbstractArray{<:Point})
  n = get_nfields(fpt)
  f = testitem(fpt)
  cb,cf = return_cache(f,x)
  ca = PTArray(fill(cb.array,n))
  ca,cb,cf
end

function Arrays.evaluate!(cache,fpt::GenericPTField,x::AbstractArray{<:Point})
  ca,c... = cache
  p,t = get_params(fpt),get_times(fpt)
  np = length(p)
  @inbounds for q = eachindex(ca)
    pq = p[fast_idx(q,np)]
    tq = t[slow_idx(q,np)]
    fq = get_field(fpt,pq,tq)
    ca[q] = evaluate!(c,fq,x)
  end
  ca
end

struct PTSingleFieldFEFunction{T<:CellField} <: FEFunction
  cell_field::T
  cell_dof_values::PTArray{<:AbstractVector{<:Number}}
  free_values::PTArray{<:Number}
  dirichlet_values::PTArray{<:Number}
  fe_space::SingleFieldFESpace
end

get_data(f::PTSingleFieldFEFunction) = get_data(f.cell_field)
get_triangulation(f::PTSingleFieldFEFunction) = get_triangulation(f.cell_field)
DomainStyle(::Type{PTSingleFieldFEFunction{T}}) where T = DomainStyle(T)

get_free_dof_values(f::PTSingleFieldFEFunction) = f.free_values
get_cell_dof_values(f::PTSingleFieldFEFunction) = f.cell_dof_values
get_fe_space(f::PTSingleFieldFEFunction) = f.fe_space

const PTSingleFieldTypes = Union{GenericCellField,PTSingleFieldFEFunction}

function TransientCellField(single_field::PTArray,derivatives::Tuple)
  TransientSingleFieldCellField(single_field,derivatives)
end

struct PTMultiFieldFEFunction{T<:MultiFieldCellField} <: FEFunction
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
