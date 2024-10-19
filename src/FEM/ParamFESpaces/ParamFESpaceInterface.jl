function get_dirichlet_cells end
get_dirichlet_cells(f::FESpace) = @abstractmethod
get_dirichlet_cells(f::UnconstrainedFESpace) = f.dirichlet_cells
get_dirichlet_cells(f::TProductFESpace) = get_dirichlet_cells(f.space)

ParamDataStructures.param_length(f::FESpace) = 0

"""
    abstract type SingleFieldParamFESpace <: SingleFieldFESpace end

Parametric extension of a [`SingleFieldFESpace`](@ref) in [`Gridap`](@ref). The
FE spaces inhereting are (trial) spaces on which we can easily define a
[`ParamFEFunction`](@ref). Most commonly, a SingleFieldParamFESpace is
characterized by parametric Dirichlet boundary conditions, but a standard
nonparametric DBC can be prescribed on such spaces.


Subtypes:
- TrivialParamFESpace{S,L} <: SingleFieldParamFESpace
- TrialParamFESpace{S} <: SingleFieldParamFESpace

"""
abstract type SingleFieldParamFESpace <: SingleFieldFESpace end

FESpaces.get_free_dof_ids(f::SingleFieldParamFESpace) = get_free_dof_ids(f.space)

FESpaces.get_triangulation(f::SingleFieldParamFESpace) = get_triangulation(f.space)

FESpaces.get_dof_value_type(f::SingleFieldParamFESpace) = get_dof_value_type(f.space)

FESpaces.get_cell_dof_ids(f::SingleFieldParamFESpace) = get_cell_dof_ids(f.space)

FESpaces.get_fe_basis(f::SingleFieldParamFESpace) = get_fe_basis(f.space)

FESpaces.get_trial_fe_basis(f::SingleFieldParamFESpace) = get_trial_fe_basis(f.space)

FESpaces.get_fe_dof_basis(f::SingleFieldParamFESpace) = get_fe_dof_basis(f.space)

FESpaces.get_cell_isconstrained(f::SingleFieldParamFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::SingleFieldParamFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::SingleFieldParamFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::SingleFieldParamFESpace) = get_cell_is_dirichlet(f.space)

FESpaces.num_dirichlet_dofs(f::SingleFieldParamFESpace) = num_dirichlet_dofs(f.space)

FESpaces.num_dirichlet_tags(f::SingleFieldParamFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::SingleFieldParamFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.scatter_free_and_dirichlet_values(f::SingleFieldParamFESpace,fv,dv) = scatter_free_and_dirichlet_values(f.space,fv,dv)

function get_vector_index_map(f::SingleFieldParamFESpace)
  get_vector_index_map(f.space)
end

function get_matrix_index_map(f::SingleFieldParamFESpace,g::SingleFieldFESpace)
  get_matrix_index_map(f.space,g)
end

get_dirichlet_cells(f::SingleFieldParamFESpace) = get_dirichlet_cells(f.space)

# These functions allow us to use global ParamArrays

function FESpaces.get_vector_type(f::SingleFieldParamFESpace)
  V = get_vector_type(f.space)
  L = param_length(f)
  typeof(param_array(V(),L))
end

function FESpaces.gather_free_and_dirichlet_values!(
  free_vals,
  dirichlet_vals,
  f::SingleFieldParamFESpace,
  cell_vals)

  cell_dofs = get_cell_dof_ids(f)
  cache_vals = array_cache(cell_vals)
  cache_dofs = array_cache(cell_dofs)
  cells = 1:length(cell_vals)

  FESpaces._free_and_dirichlet_values_fill!(
    free_vals,
    dirichlet_vals,
    cache_vals,
    cache_dofs,
    cell_vals,
    cell_dofs,
    cells)

  (free_vals,dirichlet_vals)
end

function FESpaces.gather_dirichlet_values!(
  dirichlet_vals,
  f::SingleFieldParamFESpace,
  cell_vals)

  cell_dofs = get_cell_dof_ids(f)
  cache_vals = array_cache(cell_vals)
  cache_dofs = array_cache(cell_dofs)
  free_vals = zero_free_values(f)
  cells = get_dirichlet_cells(f)

  FESpaces._free_and_dirichlet_values_fill!(
    free_vals,
    dirichlet_vals,
    cache_vals,
    cache_dofs,
    cell_vals,
    cell_dofs,
    cells)

  dirichlet_vals
end

function FESpaces._fill_dirichlet_values_for_tag!(
  dirichlet_values::AbstractParamVector,
  dv::AbstractParamVector,
  tag,
  dirichlet_dof_to_tag)

  @check param_length(dirichlet_values) == param_length(dv)
  for dof in 1:innerlength(dv)
    if dirichlet_dof_to_tag[dof] == tag
      @inbounds for k in param_eachindex(dirichlet_values)
        dirichlet_values[k][dof] = dv[k][dof]
      end
    end
  end
end

function FESpaces._free_and_dirichlet_values_fill!(
  free_vals::AbstractParamVector,
  dirichlet_vals::AbstractParamVector,
  cache_vals,
  cache_dofs,
  cell_vals,
  cell_dofs,
  cells)

  @check param_length(free_vals) == param_length(dirichlet_vals)
  for cell in cells
    vals = getindex!(cache_vals,cell_vals,cell)
    dofs = getindex!(cache_dofs,cell_dofs,cell)
    for (i,dof) in enumerate(dofs)
      @inbounds for k in param_eachindex(dirichlet_vals)
        val = vals[k][i]
        if dof > 0
          free_vals[k][dofs] = val
        elseif dof < 0
          dirichlet_vals[k][-dof] = val
        else
          @unreachable "dof ids either positive or negative, not zero"
        end
      end
    end
  end
end

"""
    TrivialParamFESpace{S,L} <: SingleFieldParamFESpace

Wrapper for nonparametric FE spaces that we wish assumed a parametric length `L`

"""

struct TrivialParamFESpace{S,L} <: SingleFieldParamFESpace
  space::S
  TrivialParamFESpace(space::S,::Val{L}) where {S,L} = new{S,L}(space)
end

TrivialParamFESpace(space::FESpace,plength::Integer) = TrivialParamFESpace(space,Val{plength}())

ParamDataStructures.param_length(f::TrivialParamFESpace{S,L}) where {S,L} = L
ParamDataStructures.to_param_quantity(f::SingleFieldParamFESpace,plength::Integer) = f
ParamDataStructures.to_param_quantity(f::SingleFieldFESpace,plength::Integer) = TrivialParamFESpace(f,Val{plength}())
ParamDataStructures.param_getindex(f::TrivialParamFESpace,index::Integer) = f.space

FESpaces.ConstraintStyle(::Type{<:TrivialParamFESpace{S}}) where S = ConstraintStyle(S)

# Extend some of Gridap's functions when needed
function FESpaces.FEFunction(
  f::TrivialParamFESpace{<:ZeroMeanFESpace},
  free_values::AbstractParamVector,
  dirichlet_values::AbstractParamVector)

  zf = f.space
  c = FESpaces._compute_new_fixedval(
    free_values,
    dirichlet_values,
    f.vol_i,
    f.vol,
    f.space.dof_to_fix
  )
  fv = free_values .+ c
  dv = dirichlet_values .+ c
  FEFunction(f.space,fv,dv)
end

function FESpaces._compute_new_fixedval(
  fv::AbstractParamVector,
  dv::AbstractParamVector,
  vol_i,
  vol,
  fixed_dof)

  @assert innerlength(fv) + 1 == length(vol_i)
  @assert innerlength(dv) == 1
  @assert param_length(fv) == param_length(dv)

  c = zero(eltype(vol_i),param_length(fv))
  @inbounds for k in param_eachindex(fv)
    ck = c[k]
    fvk = fv[k]
    dvk = dv[k]
    for i=1:fixed_dof-1
      ck += fvk[i]*vol_i[i]
    end
    ck += first(dvk)*vol_i[fixed_dof]
    for i=fixed_dof+1:length(vol_i)
      ck += fvk[i-1]*vol_i[i]
    end
    ck = -ck/vol
    c[k] = ck
  end

  return c
end

function FESpaces.EvaluationFunction(f::TrivialParamFESpace{<:ZeroMeanFESpace},free_values)
  zf = f.space
  f′ = TrivialParamFESpace(zf.space,param_length(f))
  FEFunction(f′,free_values)
end

function FESpaces.EvaluationFunction(f::TrivialParamFESpace{<:TrialFESpace{<:ZeroMeanFESpace}},free_values)
  zf = f.space
  f′ = TrivialParamFESpace(zf.space,param_length(f))
  FEFunction(f′,free_values)
end

function FESpaces.EvaluationFunction(f::TrivialParamFESpace{<:TrivialParamFESpace{<:ZeroMeanFESpace}},free_values)
  @error "do i need this?"
end

for T in (:TrialFESpace,:TProductFESpace,:ZeroMeanFESpace)
  @eval begin
    function FESpaces.scatter_free_and_dirichlet_values(
      tf::TrivialParamFESpace{<:$T},
      fv::AbstractParamVector,
      dv::AbstractParamVector)

      f = tf.space
      tf′ = TrivialParamFESpace(f.space,param_length(tf))
      scatter_free_and_dirichlet_values(tf′,fv,dv)
    end
  end
end

function FESpaces.scatter_free_and_dirichlet_values(
  tf::TrivialParamFESpace{<:FESpaceWithConstantFixed{T}},
  fv::AbstractParamVector,
  dv::AbstractParamVector
  ) where T<:FESpaces.FixConstant

  @assert innerlength(dv) == 1
  f = tf.space
  tf′ = TrivialParamFESpace(f.space,param_length(f))
  _dv = similar(dv,eltype(dv),0)
  _fv = VectorWithEntryInserted(fv,f.dof_to_fix,get_param_entry(dv,1))
  scatter_free_and_dirichlet_values(tf′,_fv,_dv)
end

function FESpaces.scatter_free_and_dirichlet_values(
  tf::TrivialParamFESpace{<:FESpaceWithConstantFixed{T}},
  fv::AbstractParamVector,
  dv::AbstractParamVector
  ) where T<:FESpaces.DoNotFixConstant

  @assert innerlength(dv) == 0
  f = tf.space
  tf′ = TrivialParamFESpace(f.space,param_length(f))
  scatter_free_and_dirichlet_values(tf′,fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values(
  tf::TrivialParamFESpace{<:FESpaceWithConstantFixed{FixConstant}},
  cv)

  f = tf.space
  tf′ = TrivialParamFESpace(f.space,param_length(f))
  _fv,_dv = gather_free_and_dirichlet_values(tf′,cv)
  @assert innerlength(_dv) == 0
  fv = ParamVectorWithEntryRemoved(_fv,f.dof_to_fix)
  dv = get_param_entry(_fv,f.dof_to_fix:f.dof_to_fix)
  (fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values(
  tf::TrivialParamFESpace{<:FESpaceWithConstantFixed{DoNotFixConstant}},
  cv)

  f = tf.space
  tf′ = TrivialParamFESpace(f.space,param_length(f))
  gather_free_and_dirichlet_values(tf′,cv)
end

function FESpaces.gather_free_and_dirichlet_values!(
  fv,
  dv,
  tf::TrivialParamFESpace{<:FESpaceWithConstantFixed{FixConstant}},
  cv)

  @assert innerlength(dv) == 1
  f = tf.space
  tf′ = TrivialParamFESpace(f.space,param_length(f))
  _dv = similar(dv,eltype(dv),0)
  _fv = VectorWithEntryInserted(fv,f.dof_to_fix,zero(eltype(fv)))
  gather_free_and_dirichlet_values!(_fv,_dv,tf′,cv)
  dv[1] = _fv.value
  (fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values!(
  fv,
  dv,
  tf::TrivialParamFESpace{<:FESpaceWithConstantFixed{DoNotFixConstant}},
  cv)

  f = tf.space
  tf′ = TrivialParamFESpace(f.space,param_length(f))
  gather_free_and_dirichlet_values!(fv,dv,tf′,cv)
end

function FESpaces.TrialFESpace(
  f::TrivialParamFESpace{<:FESpaceWithConstantFixed{CA}}
  ) where CA

  f = tf.space
  U = TrialFESpace(f)
  TrivialParamFESpace(U,param_length(f))
end

# utils

struct ParamVectorWithEntryRemoved{T,L,A} <: ParamVector{T,L}
  a::A
  index::Int
  function ParamVectorWithEntryRemoved(a::A,index::Integer) where {T,L,A<:AbstractParamVector{T,L}}
    @assert 1 <= index <= innerlength(a)
    new{T,L,A}(a,index)
  end
end

function Arrays.VectorWithEntryRemoved(a::AbstractParamVector,index::Int)
  ParamVectorWithEntryRemoved(a,index)
end

const ConsecPVWithEntryRemoved{T,L} = ParamVectorWithEntryRemoved{T,L,ConsecutiveParamVector{T,L}}

Base.size(v::ParamVectorWithEntryRemoved) = size(v.a)

function Base.getindex(v::ParamVectorWithEntryRemoved,i::Integer)
  i < v.index ? get_param_entry(v.a,i) : get_param_entry(v.a,i+1)
end

function Base.setindex!(a::ConsecPVWithEntryRemoved,v,i::Integer)
  i < a.index ? (get_all_data(a.a)[i,:] = v) : (get_all_data(a.a)[i+1,:] = v)
end

function Base.setindex!(a::ParamVectorWithEntryRemoved,v,i::Integer)
  @inbounds for k in param_eachindex(a.a)
    if i < a.index
      a.a[k][i] = v
    else
      a.a[k][i+1] = v
    end
  end
end

function Base.sum(v::ConsecPVWithEntryRemoved)
  data = get_all_data(v.a)
  sum(data,dims=1) - data[v.index,:]
end

function Base.sum(v::ParamVectorWithEntryRemoved{T,L})
  s = zeros(T,L)
  @inbounds for k in param_eachindex(v)
    ak = v.a[k]
    for i in 1:innerlength(v)
      if i < v.index
        s[i] += ak[i]
      else
        s[i] += ak[i+1]
      end
    end
  end
end

struct ParamVectorWithEntryInserted{T,L,A} <: ParamVector{T,L}
  a::A
  index::Int
  value::Vector{T}
  function ParamVectorWithEntryInserted(a::A,index::Integer,value::Vector{T}) where {T,L,A<:AbstractParamVector{T,L}}
    @assert 1 <= index <= innerlength(a)
    new{T,L,A}(a,index,value)
  end
end

function Arrays.VectorWithEntryInserted(a::AbstractParamVector,index::Int,value::AbstractVector)
  ParamVectorWithEntryInserted(a,index,value)
end

const ConsecPVWithEntryInserted{T,L} = ParamVectorWithEntryInserted{T,L,ConsecutiveParamVector{T,L}}

Base.size(v::ParamVectorWithEntryInserted) = size(v.a)

function Base.getindex(v::ParamVectorWithEntryInserted,i::Integer)
  i < v.index ? get_param_entry(v.a,i) : (i == v.index ? v.value : get_param_entry(v.a,i-1))
end

function Base.setindex!(a::ConsecPVWithEntryInserted,v,i::Integer)
  i < a.index ? (get_all_data(a)[i,:] = v) : (i == a.index ? a.value : get_all_data(a)[i-1,:] = v)
end

function Base.setindex!(a::ParamVectorWithEntryInserted,v,i::Integer)
  @inbounds for k in param_eachindex(a.a)
    if i < a.index
      a.a[k][i] = v
    elseif i == a.index
      a.value
    else
      a.a[k][i+1] = v
    end
  end
end

function Base.sum(v::ConsecPVWithEntryInserted)
  data = get_all_data(v.a)
  sum(data,dims=1) + v.value
end

function Base.sum(v::ParamVectorWithEntryInserted{T,L})
  s = zeros(T,L)
  @inbounds for k in param_eachindex(v.a)
    ak = v.a[k]
    for i in 1:innerlength(v.a)
      if i < v.index
        s[i] += ak[i]
      elseif i == v.index
        s[i] += v.value[i]
      else
        s[i] += ak[i+1]
      end
    end
  end
end

# for testing purposes

function FESpaces.test_single_field_fe_space(f::SingleFieldParamFESpace,pred=(==))
  fe_basis = get_fe_basis(f)
  @test isa(fe_basis,CellField)
  test_fe_space(f)
  dirichlet_values = zero_dirichlet_values(f)
  @test length(dirichlet_values) == num_dirichlet_dofs(f)
  free_values = zero_free_values(f)
  cell_vals = scatter_free_and_dirichlet_values(f,free_values,dirichlet_values)
  fv, dv = gather_free_and_dirichlet_values(f,cell_vals)
  @test pred(fv,free_values)
  @test pred(dv,dirichlet_values)
  gather_free_and_dirichlet_values!(fv,dv,f,cell_vals)
  @test pred(fv,free_values)
  @test pred(dv,dirichlet_values)
  fv, dv = gather_free_and_dirichlet_values!(fv,dv,f,cell_vals)
  @test pred(fv,free_values)
  @test pred(dv,dirichlet_values)
  fe_function = FEFunction(f,free_values,dirichlet_values)
  @test isa(fe_function,SingleFieldParamFEFunction)
  test_fe_function(fe_function)
  ddof_to_tag = get_dirichlet_dof_tag(f)
  @test length(ddof_to_tag) == num_dirichlet_dofs(f)
  if length(get_dirichlet_dof_tag(f)) != 0
    @test maximum(get_dirichlet_dof_tag(f)) <= num_dirichlet_tags(f)
  end
  cell_dof_basis = get_fe_dof_basis(f)
  @test isa(cell_dof_basis,CellDof)
end
