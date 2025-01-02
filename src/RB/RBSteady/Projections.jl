"""
    abstract type Projection <: Map end

Represents a basis for a `n`-dimensional vector subspace of a `N`-dimensional
vector space (where `N` ≫ `n`), to be used as a (Petrov)-Galerkin projection
operator. The kernel of a Projection is `n`-dimensional, whereas its image is
`N`-dimensional.

Subtypes:

- [`PODBasis`](@ref)
- [`TTSVDCores`](@ref)
- [`BlockProjection`](@ref)
- [`InvProjection`](@ref)
- [`ReducedProjection`](@ref)
- [`HyperReduction`](@ref)
"""
abstract type Projection <: Map end

"""
    get_basis(a::Projection) -> AbstractMatrix

Returns the basis spanning the reduced subspace represented by the projection `a`
"""
get_basis(a::Projection) = @abstractmethod

"""
    num_fe_dofs(a::Projection) -> Int

For a projection map `a` from a low dimensional space `n` to a high dimensional
one `N`, returns `N`
"""
num_fe_dofs(a::Projection) = @abstractmethod

"""
    num_reduced_dofs(a::Projection) -> Int

For a projection map `a` from a low dimensional space `n` to a high dimensional
one `N`, returns `n`
"""
num_reduced_dofs(a::Projection) = @abstractmethod

"""
    project(a::Projection,x::AbstractArray) -> AbstractArray

Projects a high-dimensional object `x` onto the subspace represented by `a`
"""
project(a::Projection,x::AbstractArray) = @abstractmethod

"""
    inv_project(a::Projection,x::AbstractArray) -> AbstractArray

Recasts a low-dimensional object `x` onto the high-dimensional space in which `a`
is immersed
"""
inv_project(a::Projection,x::AbstractArray) = @abstractmethod

"""
    galerkin_projection(a::Projection,b::Projection) -> ReducedProjection
    galerkin_projection(a::Projection,b::Projection,c::Projection,args...) -> ReducedProjection

(Petrov) Galerkin projection of a projection map `b` onto the subspace `a` (row
projection) and, if applicable, onto the subspace `c` (column projection)
"""
galerkin_projection(a::Projection,b::Projection) = @abstractmethod
galerkin_projection(a::Projection,b::Projection,c::Projection,args...) = @abstractmethod

"""
    empirical_interpolation(a::Projection) -> (AbstractVector,AbstractMatrix)

Computes the EIM of `a`. The outputs are:

- a vector of integers `i`, corresponding to a list of interpolation row indices
- a matrix `Φi = view(Φ,i)`, where `Φ = get_basis(a)`. This quantity represents
the restricted basis on the set of interpolation rows `i`
"""
empirical_interpolation(a::Projection) = @abstractmethod

rescale(op::Function,x::AbstractArray,b::Projection) = @abstractmethod

"""
    union_bases(a::Projection,b::Projection,args...) -> Projection

Computes the projection corresponding to the union of `a` and `b`. In essence this
operation performs as

Φa = get_basis(a)
Φb = get_basis(b)
Φab = union(Φa,Φb)
return gram_schmidt(Φab)
"""
union_bases(a::Projection,b::Projection,args...) = @abstractmethod

gram_schmidt(a::Projection,b::Projection,args...) = gram_schmidt(get_basis(a),get_basis(b),args...)

Base.:+(a::Projection,b::Projection) = union_bases(a,b)
Base.:-(a::Projection,b::Projection) = union_bases(a,b)
Base.:*(a::Projection,b::Projection) = galerkin_projection(a,b)
Base.:*(a::Projection,b::Projection,c::Projection) = galerkin_projection(a,b,c)
Base.:*(a::Projection,x::AbstractArray) = inv_project(a,x)
Base.:*(x::AbstractArray,b::Projection) = rescale(*,x,b)

function Base.:*(b::Projection,y::ConsecutiveParamArray)
  item = zeros(num_reduced_dofs(b))
  plength = param_length(y)
  x = consecutive_param_array(item,plength)
  mul!(x,b,y)
end

function LinearAlgebra.mul!(x::AbstractArray,b::Projection,y::AbstractArray,α::Number,β::Number)
  mul!(x,get_basis(b),y,α,β)
end

function LinearAlgebra.mul!(x::ConsecutiveParamArray,b::Projection,y::ConsecutiveParamArray,α::Number,β::Number)
  mul!(x.data,get_basis(b),y.data,α,β)
end

function project(a::Projection,x::AbstractVector)
  basis = get_basis(a)
  x̂ = basis'*x
  return x̂
end

function inv_project(a::Projection,x̂::AbstractVector)
  basis = get_basis(a)
  x = basis*x̂
  return x
end

function Arrays.return_cache(::typeof(project),a::Projection,x::AbstractVector)
  x̂ = zeros(eltype(x),num_reduced_dofs(a))
  return x̂
end

function Arrays.return_cache(::typeof(inv_project),a::Projection,x̂::AbstractVector)
  x = zeros(eltype(x̂),num_fe_dofs(a))
  return x
end

"""
    struct InvProjection <: Projection
      projection::Projection
    end

Represents the inverse map of a [`Projection`](@ref) `projection`
"""
struct InvProjection <: Projection
  projection::Projection
end

Base.adjoint(a::Projection) = InvProjection(a)

get_basis(a::InvProjection) = get_basis(a)
num_fe_dofs(a::InvProjection) = num_fe_dofs(a)
num_reduced_dofs(a::InvProjection) = num_reduced_dofs(a)
project(a::InvProjection,x::AbstractArray) = inv_project(a.projection,x)
inv_project(a::InvProjection,x::AbstractArray) = project(a.projection,x)

"""
    abstract type ReducedProjection{A<:AbstractArray} <: Projection end

Type representing a (Petrov-)Galerkin projection of a [`Projection`](@ref) onto
a reduced subspace represented by another [`Projection`](@ref).

Subtypes:

- [`ReducedAlgebraicProjection](@ref)
"""
abstract type ReducedProjection{A<:AbstractArray} <: Projection end

const ReducedVecProjection = ReducedProjection{<:AbstractMatrix}
const ReducedMatProjection = ReducedProjection{<:AbstractArray{T,3} where T}

function project(a::ReducedProjection,x::AbstractVector)
  @notimplemented
end

function inv_project(a::ReducedMatProjection,x̂::AbstractVector)
  basis = get_basis(a)
  x = contraction(basis,x̂)
  return x
end

function LinearAlgebra.mul!(
  x::AbstractArray,
  b::ReducedMatProjection,
  y::AbstractArray,
  α::Number,β::Number)

  contraction!(x,get_basis(b),y,α,β)
end

function LinearAlgebra.mul!(
  x::ConsecutiveParamArray,
  b::ReducedMatProjection,
  y::ConsecutiveParamArray,
  α::Number,β::Number)

  contraction!(x.data,get_basis(b),y.data,α,β)
end

"""
"""
struct ReducedAlgebraicProjection{A} <: ReducedProjection{A}
  basis::A
end

ReducedProjection(basis::AbstractArray) = ReducedAlgebraicProjection(basis)

get_basis(a::ReducedAlgebraicProjection) = a.basis
num_reduced_dofs(a::ReducedAlgebraicProjection) = size(get_basis(a),2)
num_reduced_dofs_left_projector(a::ReducedAlgebraicProjection) = size(get_basis(a),1)
num_reduced_dofs_right_projector(a::ReducedMatProjection) = size(get_basis(a),3)

function projection(red::AffineReduction,s::AbstractMatrix,args...)
  podred = PODReduction(ReductionStyle(red),NormStyle(red))
  projection(podred,s,args...)
end

"""
    struct PODBasis <: Projection
      basis::AbstractMatrix
    end

Projection stemming from a truncated proper orthogonal decomposition [`tpod`](@ref)
"""
struct PODBasis <: Projection
  basis::AbstractMatrix
end

function projection(red::PODReduction,s::AbstractArray{<:Number},args...)
  basis = reduction(red,s,args...)
  PODBasis(basis)
end

function projection(red::PODReduction,s::SparseSnapshots,args...)
  basis = reduction(red,s,args...)
  basis′ = recast(basis,s)
  PODBasis(basis′)
end

get_basis(a::PODBasis) = a.basis
num_fe_dofs(a::PODBasis) = size(get_basis(a),1)
num_reduced_dofs(a::PODBasis) = size(get_basis(a),2)

union_bases(a::PODBasis,b::PODBasis,args...) = union_bases(a,get_basis(b),args...)

function union_bases(a::PODBasis,basis_b::AbstractMatrix,args...)
  basis_a = get_basis(a)
  basis_ab, = gram_schmidt(basis_b,basis_a,args...)
  PODBasis(basis_ab)
end

function galerkin_projection(proj_left::PODBasis,a::PODBasis)
  basis_left = get_basis(proj_left)
  basis = get_basis(a)
  proj_basis = galerkin_projection(basis_left,basis)
  return ReducedProjection(proj_basis)
end

function galerkin_projection(proj_left::PODBasis,a::PODBasis,proj_right::PODBasis)
  basis_left = get_basis(proj_left)
  basis = get_basis(a)
  basis_right = get_basis(proj_right)
  proj_basis = galerkin_projection(basis_left,basis,basis_right)
  return ReducedProjection(proj_basis)
end

function empirical_interpolation(a::PODBasis)
  empirical_interpolation(get_basis(a))
end

function rescale(op::Function,x::AbstractArray,b::PODBasis)
  PODBasis(op(x,get_basis(b)))
end

# TT interface

"""
    struct TTSVDCores <: Projection
      cores::AbstractVector{<:AbstractArray{T,3} where T}
      dof_map::AbstractDofMap
    end

Projection stemming from a tensor train SVD [`ttsvd`](@ref). For reindexing purposes
a field `dof_map` is provided along with the tensor train cores `cores`
"""
struct TTSVDCores <: Projection
  cores::AbstractVector{<:AbstractArray{T,3} where T}
  dof_map::AbstractDofMap
end

function projection(red::TTSVDReduction,s::AbstractArray{<:Number},args...)
  cores = reduction(red,s,args...)
  dof_map = get_dof_map(s)
  TTSVDCores(cores,dof_map)
end

function projection(red::TTSVDReduction,s::SparseSnapshots,args...)
  cores = reduction(red,s,args...)
  cores′ = recast(cores,s)
  dof_map = get_dof_map(s)
  TTSVDCores(cores′,dof_map)
end

get_cores(a::Projection) = @notimplemented
get_cores(a::TTSVDCores) = a.cores

get_basis(a::TTSVDCores) = cores2basis(get_dof_map(a),get_cores(a)...)
num_fe_dofs(a::TTSVDCores) = prod(map(c -> size(c,2),get_cores(a)))
num_reduced_dofs(a::TTSVDCores) = size(last(get_cores(a)),3)

DofMaps.get_dof_map(a::TTSVDCores) = a.dof_map

function union_bases(a::TTSVDCores,b::TTSVDCores,args...)
  @check get_dof_map(a) == get_dof_map(b)
  union_bases(a,get_cores(b),args...)
end

function union_bases(
  a::TTSVDCores,
  cores_b::AbstractVector{<:AbstractArray},
  args...
  )

  red_style = TTSVDReduction(fill(1e-4,length(cores_a)))
  union_bases(a,cores_b,red_style,args...)
end

function union_bases(
  a::TTSVDCores,
  cores_b::AbstractVector{<:AbstractArray},
  red_style::ReductionStyle,
  args...
  )

  cores_a = get_cores(a)
  @check length(cores_a) == length(cores_b)

  cores_ab = block_cores([cores_a,cores_b])
  orthogonalize!(red_style,cores_ab,args...)
  TTSVDCores(cores_ab,get_dof_map(a))
end

function galerkin_projection(proj_left::TTSVDCores,a::TTSVDCores)
  cores_left = get_cores(proj_left)
  cores = get_cores(a)
  proj_basis = galerkin_projection(cores_left,cores)
  return ReducedProjection(proj_basis)
end

function galerkin_projection(proj_left::TTSVDCores,a::TTSVDCores,proj_right::TTSVDCores)
  cores_left = get_cores(proj_left)
  cores = get_cores(a)
  cores_right = get_cores(proj_right)
  proj_basis = galerkin_projection(cores_left,cores,cores_right)
  return ReducedProjection(proj_basis)
end

function empirical_interpolation(a::TTSVDCores)
  local indices,interp
  cores = get_cores(a)
  dof_map = get_dof_map(a)
  c = cores2basis(first(cores))
  cache = eim_cache(c)
  vinds = Vector{Int32}[]
  for i = eachindex(cores)
    inds,interp = empirical_interpolation!(cache,c)
    push!(vinds,copy(inds))
    if i < length(cores)
      interp_core = reshape(interp,1,size(interp)...)
      c = cores2basis(interp_core,cores[i+1])
    else
      indices = get_basis_indices(vinds,dof_map)
    end
  end
  return indices,interp
end

function rescale(op::Function,x::AbstractRankTensor{D},b::TTSVDCores) where D
  cores = get_cores(b)
  dof_map = get_dof_map(b)
  if D == ndims(dof_map)
    TTSVDCores(op(x,cores),dof_map)
  else
    c1 = op(x,cores[1:D])
    c2 = cores[D+1:end]
    TTSVDCores([c1...,c2...],dof_map)
  end
end

# multi field interface

function Arrays.return_type(::typeof(projection),::PODReduction,s::Snapshots{T}) where T
  PODBasis
end

function Arrays.return_type(::typeof(projection),::TTSVDReduction,s::Snapshots)
  TTSVDCores
end

function Arrays.return_cache(::typeof(projection),red::Reduction,s::BlockSnapshots)
  i = findfirst(s.touched)
  @notimplementedif isnothing(i)
  A = return_type(projection,red,blocks(s)[i])
  block_basis = Array{A,ndims(s)}(undef,size(s))
  touched = s.touched
  return BlockProjection(block_basis,touched)
end

function projection(red::Reduction,s::BlockSnapshots)
  basis = return_cache(projection,red,s)
  for i in eachindex(basis)
    if basis.touched[i]
      basis[i] = projection(red,s[i])
    end
  end
  return basis
end

function projection(red::Reduction,s::BlockSnapshots,norm_matrix)
  basis = return_cache(projection,red,s)
  for i in eachindex(basis)
    if basis.touched[i]
      basis[i] = projection(red,s[i],norm_matrix[Block(i,i)])
    end
  end
  return basis
end

"""
    struct BlockProjection{A,N} <: Projection end

Block container for Projection of type `A` in a MultiField setting. This
type is conceived similarly to [`ArrayBlock`](@ref) in [`Gridap`](@ref)
"""
struct BlockProjection{A<:Projection,N} <: Projection
  array::Array{A,N}
  touched::Array{Bool,N}

  function BlockProjection(
    array::Array{A,N},
    touched::Array{Bool,N}
    ) where {A<:Projection,N}

    @check size(array) == size(touched)
    new{A,N}(array,touched)
  end
end

function BlockProjection(a::AbstractArray{A},touched::Array{Bool,N}) where {A<:Projection,N}
  array = Array{A,N}(undef,k.size)
  for i in touched
    if touched[i]
      array[i] = a[i]
    end
  end
  BlockProjection(array,touched)
end

Base.ndims(a::BlockProjection) = ndims(a.touched)
Base.size(a::BlockProjection) = size(a.touched)
Base.axes(a::BlockProjection) = axes(a.touched)
Base.eachindex(a::BlockProjection) = eachindex(a.touched)

function Base.getindex(a::BlockProjection,i...)
  if !a.touched[i...]
    return nothing
  end
  a.array[i...]
end

function Base.setindex!(a::BlockProjection,v,i...)
  @check a.touched[i...] "Only touched entries can be set"
  a.array[i...] = v
end

function num_fe_dofs(a::BlockProjection)
  dofs = zeros(Int,length(a))
  for i in eachindex(a)
    if a.touched[i]
      dofs[i] = num_fe_dofs(a[i])
    end
  end
  return dofs
end

function num_reduced_dofs(a::BlockProjection)
  dofs = zeros(Int,size(a))
  for i in eachindex(a)
    if a.touched[i]
      dofs[i] = num_reduced_dofs(a[i])
    end
  end
  return dofs
end

for f in (:project,:inv_project)
  @eval begin
    function Arrays.return_cache(
      ::typeof($f),
      a::BlockProjection,
      x::Union{BlockVector,BlockParamVector})

      @check size(a) == nblocks(x)
      y = Vector{eltype(x)}(undef,nblocks(a))
      for i in eachindex(a)
        if a.touched[i]
          y[Block(i)] = return_cache($f,a[i],blocks(x)[i])
        else
          y[Block(i)] = blocks(x)[i]
        end
      end
      return mortar(y)
    end

    function $f(a::BlockProjection,x::Union{BlockArray,BlockParamArray})
      y = return_cache($f,a,x)
      for i in eachindex(a)
        if a.touched[i]
          y[Block(i)] = $f(a[i],blocks(x)[i])
        end
      end
      return y
    end
  end
end

"""
    enrich!(red::SupremizerReduction,a::BlockProjection,norm_matrix,supr_matrix) -> Nothing

In-place augmentation of the primal block of a [`BlockProjection`](@ref) `a`.
This function has the purpose of stabilizing the reduced equations stemming from
a saddle point problem
"""
function enrich!(
  red::SupremizerReduction,
  a::BlockProjection,
  norm_matrix::BlockMatrix,
  supr_matrix::BlockMatrix)

  @check a.touched[1] "Primal field not defined"
  red_style = ReductionStyle(red)
  a_primal,a_dual... = a.array
  X_primal = norm_matrix[Block(1,1)]
  H_primal = cholesky(X_primal)
  for i = eachindex(a_dual)
    dual_i = get_basis(a_dual[i])
    C_primal_dual_i = supr_matrix[Block(1,i+1)]
    supr_i = H_primal \ C_primal_dual_i * dual_i
    a_primal = union_bases(a_primal,supr_i,X_primal,red_style)
  end
  a[1] = a_primal
  return
end

function enrich!(
  red::SupremizerReduction{<:TTSVDRanks},
  a::BlockProjection,
  norm_matrix::BlockRankTensor,
  supr_matrix::BlockRankTensor)

  @check a.touched[1] "Primal field not defined"
  red_style = ReductionStyle(red)
  a_primal,a_dual... = a.array
  primal_map = get_dof_map(a_primal)
  X_primal = norm_matrix[Block(1,1)]
  for i = eachindex(a_dual)
    dual_i = get_cores(a_dual[i])
    C_primal_dual_i = supr_matrix[Block(1,i+1)]
    supr_i = tt_supremizer(X_primal,C_primal_dual_i,dual_i)
    a_primal = union_bases(a_primal,supr_i,red_style,X_primal)
  end
  a[1] = a_primal
  return
end
