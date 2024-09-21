function empirical_interpolation!(cache,A::AbstractMatrix)
  I,res = cache
  m,n = size(A)
  resize!(res,m)
  resize!(I,n)
  I[1] = argmax(abs.(A[:,1]))
  if n > 1
    @inbounds for i = 2:n
      Bi = A[:,1:i-1]
      Ci = A[I[1:i-1],1:i-1]
      Di = A[I[1:i-1],i]
      res .= A[:,i] - Bi*(Ci \ Di)
      I[i] = argmax(abs.(res))
    end
  end
  Ai = view(A,I,:)
  return I,Ai
end

function empirical_interpolation!(cache,core::AbstractArray{T,3}) where T
  @check size(C,1) == 1
  _cache...,Iv = cache
  A = dropdims(core;dims=1)
  I,Ai = empirical_interpolation!(_cache,A)
  push!(Iv,copy(I))
  return I,Ai
end

function eim_cache(A::AbstractMatrix)
  m,n = size(A)
  res = zeros(eltype(A),m)
  I = zeros(Int32,n)
  return I,res
end

function eim_cache(core::AbstractArray{T,3}) where T
  m,n = size(core,2),size(core,1)
  res = zeros(T,m)
  I = zeros(Int32,n)
  Iv = Vector{Int32}[]
  return I,res,Iv
end

function empirical_interpolation(a::AbstractArray)
  cache = eim_cache(a)
  i,ai = empirical_interpolation!(cache,a)
  return i,ai
end

function empirical_interpolation(A::ParamSparseMatrix)
  i,ai = empirical_interpolation(A.data)
  i′ = recast_indices(I,param_getindex(A,1))
  return i′,ai
end

abstract type AbstractIntegrationDomain{Ti} <: AbstractVector{Ti} end

struct IntegrationDomain <: AbstractIntegrationDomain{Int32}
  indices::Vector{Int32}
end

Base.size(i::IntegrationDomain) = size(i.indices)
Base.getindex(i::IntegrationDomain,j::Integer) = getindex(i.indices,j)

get_indices(i::IntegrationDomain) = i.indices_space
union_indices(i::IntegrationDomain...) = union(get_indices.(i)...)
function ordered_common_locations(i::IntegrationDomain,union_indices::AbstractVector)
  filter(!isnothing,indexin(i,union_indices))::Vector{<:Integer}
end

function Base.getindex(a::AbstractParamArray,i::IntegrationDomain)
  entry = zeros(eltype2(a),length(i))
  entries = array_of_consecutive_arrays(entry,param_length(a))
  for ip = param_eachindex(entries)
    for (i,is) in enumerate(indices)
      v = consecutive_getindex(a,is,ip)
      consecutive_setindex!(entries,v,i,ip)
    end
  end
  return entries
end

function Base.getindex(a::ParamSparseMatrix,i::IntegrationDomain)
  entry = zeros(eltype2(a),length(i))
  entries = array_of_consecutive_arrays(entry,param_length(a))
  for ip = param_eachindex(entries)
    for (i,is) in enumerate(indices)
      v = param_getindex(s,ip)[is]
      consecutive_setindex!(entries,v,i,ip)
    end
  end
  return entries
end

abstract type HyperReduction{A,I} <: ReducedProjection{A} end

HyperReduction(::AbstractReduction,args...) = @abstractmethod

get_interpolation(a::HyperReduction) = @abstractmethod
get_integration_domain(a::HyperReduction) = @abstractmethod

num_reduced_dofs(a::HyperReduction) = num_reduced_dofs(get_basis(a))
num_reduced_dofs_left_projector(a::HyperReduction) = num_reduced_dofs_left_projector(get_basis(a))
num_reduced_dofs_right_projector(a::HyperReduction) = num_reduced_dofs_right_projector(get_basis(a))

get_indices(a::HyperReduction) = get_indices(get_integration_domain(a))
union_indices(a::HyperReduction...) = union_indices(get_integration_domain.(a)...)

function project!(cache,a::HyperReduction,b::AbstractParamArray)
  cache = coeff,b̂
  interp = get_interpolation(a)
  ldiv!(coeff,interp,b)
  b̂ .+= a*coeff
  return b̂
end

struct MDEIM{A<:AbstractArray,I<:AbstractIntegrationDomain} <: HyperReduction{A,I}
  basis::ReducedProjection{A}
  interpolation::Factorization
  integration_domain::I
end

get_basis(a::MDEIM) = a.basis
get_interpolation(a::MDEIM) = a.interpolation
get_integration_domain(a::MDEIM) = a.integration_domain

function HyperReduction(
  red::AbstractMDEIMReduction,
  s::AbstractSnapshots,
  test::FESubspace)

  basis = projection(get_reduction(red),s)
  proj_basis = galerkin_projection(test,basis)
  indices,interp = empirical_interpolation(basis)
  factor = lu(interp)
  integration_domain = IntegrationDomain(indices)
  return MDEIM(proj_basis,factor,integration_domain)
end

function HyperReduction(
  red::AbstractMDEIMReduction,
  s::AbstractSnapshots,
  trial::FESubspace,
  test::FESubspace)

  basis = projection(get_reduction(red),s)
  proj_basis = galerkin_projection(test,basis,trial)
  indices,factor = empirical_interpolation(basis)
  factor = lu(interp)
  integration_domain = IntegrationDomain(indices)
  return MDEIM(proj_basis,factor,integration_domain)
end

function get_reduced_cells(cell_dof_ids,dofs::AbstractVector)
  cells = eltype(eltype(cell_dof_ids))[]
  for (cell,celldofs) = enumerate(cell_dof_ids)
    if !isempty(intersect(dofs,celldofs))
      append!(cells,cell)
    end
  end
  return unique(cells)
end

function get_reduced_cells(
  trian::Triangulation,
  ids::AbstractVector,
  test::FESpace)

  cell_dof_ids = get_cell_dof_ids(test,trian)
  indices_space_rows = fast_index(ids,num_free_dofs(test))
  red_integr_cells = get_reduced_cells(cell_dof_ids,indices_space_rows)
  return red_integr_cells
end

function get_reduced_cells(
  trian::Triangulation,
  ids::AbstractVector,
  trial::FESpace,
  test::FESpace)

  cell_dof_ids_trial = get_cell_dof_ids(trial,trian)
  cell_dof_ids_test = get_cell_dof_ids(test,trian)
  indices_space_cols = slow_index(ids,num_free_dofs(test))
  indices_space_rows = fast_index(ids,num_free_dofs(test))
  red_integr_cells_trial = get_reduced_cells(cell_dof_ids_trial,indices_space_cols)
  red_integr_cells_test = get_reduced_cells(cell_dof_ids_test,indices_space_rows)
  red_integr_cells = union(red_integr_cells_trial,red_integr_cells_test)
  return red_integr_cells
end

function reduced_triangulation(trian::Triangulation,b::HyperReduction,r::FESubspace...)
  f = map(get_space,r)
  indices = get_integration_domain(b)
  red_integr_cells = get_reduced_cells(trian,indices,f...)
  red_trian = view(trian,red_integr_cells)
  return red_trian
end

function Algebra.allocate_matrix(::Type{M},m::Integer,n::Integer) where M
  T = eltype(M)
  zeros(T,m,n)
end

function allocate_coefficient(a::HyperReduction,nparams::Int)
  n = num_reduced_dofs(b)
  coeffvec = allocate_vector(Vector{Float64},n)
  coeff = array_of_consecutive_arrays(coeffvec,nparams)
  return coeff
end

allocate_coefficient(a::Projection,r::AbstractRealization) = allocate_coefficient(a,num_params(r))
allocate_coefficient(a::Projection,b::AbstractParamArray) = allocate_coefficient(a,param_length(b))

function allocate_hyper_reduction(b::HyperReduction{<:ReducedVecProjection},nparams::Int)
  nrows = num_reduced_dofs_left_projector(b)
  b = allocate_vector(Vector{Float64},nrows)
  hypred = array_of_consecutive_arrays(b,nparams)
  return hypred
end

function allocate_hyper_reduction(b::HyperReduction{<:ReducedMatProjection},nparams::Int)
  nrows = num_reduced_dofs_left_projector(b)
  ncols = num_reduced_dofs_right_projector(b)
  A = allocate_matrix(Matrix{Float64},nrows,ncols)
  hypred = array_of_consecutive_arrays(A,nparams)
  return hypred
end

allocate_hyper_reduction(a::Projection,r::AbstractRealization) = allocate_hyper_reduction(a,num_params(r))
allocate_hyper_reduction(a::Projection,b::AbstractParamArray) = allocate_hyper_reduction(a,param_length(b))

function ParamDataStructures.Contribution(v::Tuple{Vararg{HyperReduction}},t::Tuple{Vararg{Triangulation}})
  AffineContribution(v,t)
end

"""
    struct AffineContribution{A,V,K} <: Contribution

The values of an AffineContribution are AffineDecompositions

"""
struct AffineContribution{A<:Projection,V,K} <: Contribution
  values::V
  trians::K
  function AffineContribution(
    values::V,
    trians::K
    ) where {A,V<:Tuple{Vararg{A}},K<:Tuple{Vararg{Triangulation}}}

    @check length(values) == length(trians)
    @check !any([t === first(trians) for t = trians[2:end]])
    new{A,V,K}(values,trians)
  end
end

function allocate_coefficient(a::AffineContribution,b::ArrayContribution)
  @check all(get_domains(a) .== get_domains(b))
  contribution(get_domains(a)) do trian
    allocate_coefficient(a[trian],b[trian])
  end
end

function allocate_hyper_reduction(a::AffineContribution,b::ArrayContribution)
  allocate_hyper_reduction(first(get_values(a)),first(get_values(b)))
end

function project!(cache,a::AffineContribution,b::ArrayContribution)
  @check length(a) == length(b)
  cache = coeff,b̂
  for (aval,bval,cval) in zip(get_values(a),get_values(b),get_values(coeff))
    project!((cval,b̂),aval,bval)
  end
  return b̂
end

function reduced_form(
  red::AbstractReduction,
  s::AbstractSnapshots,
  trian::Triangulation,
  args...)

  t = @timed begin
    hyper_red = HyperReduction(red,s,args...)
    red_trian = reduced_triangulation(trian,hyper_red,args...)
  end

  println(CostTracker(t))

  return hyper_red,red_trian
end

function reduced_residual(red::AbstractReduction,op,s::AbstractSnapshots,trian::Triangulation)
  test = get_test(op)
  reduced_form(red,s,trian,test)
end

function reduced_jacobian(red::AbstractReduction,op,s::AbstractSnapshots,trian::Triangulation)
  trial = get_trial(op)
  test = get_test(op)
  reduced_form(red,s,trian,trial,test)
end

"""
    reduced_residual(red::AbstractReduction,op::PGOperator,c::ArrayContribution)
      ) -> AffineContribution
    reduced_residual(red::AbstractReduction,op::TransientPGOperator,c::ArrayContribution)
      ) -> AffineContribution

Returns the AffineContribution corresponding to the residual snapshots stored
in the [`ArrayContribution`](@ref) `c`

"""
function reduced_residual(red::AbstractReduction,op,c::ArrayContribution)
  a,trians = map(get_domains(c),get_values(c)) do trian,values
    reduced_residual(red,op,values,trian)
  end |> tuple_of_arrays
  return Contribution(a,trians)
end

"""
    reduced_jacobian(red::AbstractReduction,op::PGOperator,c::ArrayContribution)
      ) -> AffineContribution
    reduced_jacobian(red::AbstractReduction,op::TransientPGOperator,c::TupOfArrayContribution)
      ) -> AffineContribution

Returns the AffineContribution corresponding to the jacobian snapshots stored
in the [`ArrayContribution`](@ref) `c`. In transient problems, this procedure is
run for every order of the time derivative

"""
function reduced_jacobian(red::AbstractReduction,op,c::ArrayContribution)
  a,trians = map(get_domains(c),get_values(c)) do trian,values
    reduced_jacobian(red,op,values,trian)
  end |> tuple_of_arrays
  return Contribution(a,trians)
end

function reduced_weak_form(solver::RBSolver,op,s)
  jac = jacobian_snapshots(solver,op,s)
  res = residual_snapshots(solver,op,s)
  red_jac = reduced_jacobian(get_jacobian_reduction(solver),op,jac)
  red_res = reduced_residual(get_residual_reduction(solver),op,res)
  return red_jac,red_res
end

# multi field interface

const BlockHyperReduction{A<:HyperReduction,N} = BlockProjection{A,N}

function ParamDataStructures.Contribution(
  v::Tuple{Vararg{BlockHyperReduction}},
  t::Tuple{Vararg{Triangulation}})

  AffineContribution(v,t)
end

for f in (:get_basis,:get_interpolation,:get_integration_domain)
  @eval begin
    function Arrays.return_cache(::typeof($f),a::HyperReduction)
      cache = testvalue(typeof($f(a)))
      return cache
    end

    function Arrays.return_cache(::typeof($f),a::BlockHyperReduction)
      i = findfirst(a.touched)
      @notimplementedif isempty(i)
      cache = return_cache($f,a[i])
      block_cache = Array{typeof(cache),ndims(a)}(undef,size(a))
      touched = a.touched
      return ArrayBlock(block_cache,touched)
    end

    function $f(a::BlockHyperReduction)
      cache = return_cache($f,a)
      for i in eachindex(basis)
        if cache.touched[i]
          cache[i] = $f(a[i])
        end
      end
      return cache
    end
  end
end

function Arrays.return_cache(::typeof(allocate_coefficient),a::HyperReduction,nparams::Int)
  coeffvec = testvalue(Vector{Float64})
  array_of_consecutive_arrays(coeffvec,nparams)
end

function Arrays.return_cache(::typeof(allocate_coefficient),a::BlockHyperReduction,nparams::Int)
  i = findfirst(a.touched)
  @notimplementedif isempty(i)
  coeff = return_cache(allocate_coefficient,a[i],nparams)
  block_coeff = Array{typeof(coeff),ndims(a)}(undef,size(a))
  return ArrayBlock(block_coeff,a.touched)
end

function allocate_coefficient(a::BlockHyperReduction,nparams::Int)
  coeff = return_cache(allocate_coefficient,a,nparams)
  for i in eachindex(a)
    if a.touched[i]
      coeff[i] = allocate_coefficient(a[i],nparams)
    end
  end
  return coeff
end

function Arrays.return_cache(::typeof(allocate_hyper_reduction),a::HyperReduction{<:ReducedVecProjection},nparams::Int)
  hypvec = testvalue(Vector{Float64})
  array_of_consecutive_arrays(hypvec,nparams)
end

function Arrays.return_cache(::typeof(allocate_hyper_reduction),a::HyperReduction{<:ReducedMatProjection},nparams::Int)
  hypvec = testvalue(Matrix{Float64})
  array_of_consecutive_arrays(hypvec,nparams)
end

function Arrays.return_cache(::typeof(allocate_hyper_reduction),a::BlockHyperReduction,nparams::Int)
  i = findfirst(a.touched)
  @notimplementedif isempty(i)
  hypred = return_cache(allocate_hyper_reduction,a[i],nparams)
  block_hypred = Array{typeof(hypred),ndims(a)}(undef,size(a))
  return block_hypred
end

function allocate_hyper_reduction(b::BlockHyperReduction,nparams::Int)
  hypred = return_cache(allocate_hyper_reduction,a,nparams)
  for i in eachindex(a)
    if a.touched[i]
      hypred[i] = allocate_hyper_reduction(a[i],nparams)
    end
  end
  fill_missing_blocks!(hypred)
  return mortar(hypred)
end

fill_missing_blocks!(a::AbstractArray) = @notimplemented

function fill_missing_blocks!(a::AbstractMatrix{<:AbstractMatrix{T}}) where T
  for (i,j) in Iterators.product(size(a)...)
    if isempty(a[i,j])
      row_block = findfirst(!isempty.(a[i,:]))
      col_block = findfirst(!isempty.(a[:,j]))
      @check !isnothing(row_block) "The system is ill posed"
      @check !isnothing(col_block) "The system is ill posed"
      nrows = size(row_block,1)
      ncols = size(col_block,2)
      a[i,j] = zeros(T,nrows,ncols)
    end
  end
end

function reduced_triangulation(
  trian::Triangulation,
  b::BlockHyperReduction,
  test::MultiFieldRBSpace)

  @check length(b) == num_fields(test)
  red_trian = Triangulation[]
  for i in eachindex(b)
    if b.touched[i]
      push!(red_trian,reduced_triangulation(trian,b[i],test[i]))
    end
  end
  return red_trian
end

function reduced_triangulation(
  trian::Triangulation,
  b::BlockHyperReduction,
  trial::MultiFieldRBSpace,
  test::MultiFieldRBSpace)

  @check size(b,1) == num_fields(test)
  @check size(b,2) == num_fields(trial)
  red_trian = Triangulation[]
  for (i,j) in Iterators.product(size(b)...)
    if b.touched[i,j]
      push!(red_trian,reduced_triangulation(trian,b[i,j],trial[j],test[i]))
    end
  end
  return red_trian
end

function reduced_residual(
  red::AbstractReduction,
  op,
  s::BlockSnapshots,
  trian::Triangulation)

  test = get_test(op)

  hps = HyperReduction[]
  red_trians = Triangulation[]
  for i in eachindex(s)
    if s.touched[i]
      hr,red_trian = reduced_form(red,s[i],trian,test[i])
      push!(hps,hr)
      push!(red_trians,red_trian)
    end
  end

  red_trian = ParamDataStructures.merge_triangulations(red_trians)
  hr = BlockProjection(hps,s.touched)

  return hr,red_trian
end

function reduced_jacobian(
  red::AbstractReduction,
  op,
  s::BlockSnapshots,
  trian::Triangulation)

  trial = get_trial(op)
  test = get_test(op)

  hps = HyperReduction[]
  red_trians = Triangulation[]
  for (i,j) in Iterators.product(size(s)...)
    if s.touched[i]
      hr,red_trian = reduced_form(red,s[i,j],trian,trial[j],test[i])
      push!(hps,hr)
      push!(red_trians,red_trian)
    end
  end

  red_trian = ParamDataStructures.merge_triangulations(red_trians)
  hr = BlockProjection(reshape(hps,size(s.touched)),s.touched)

  return hr,red_trian
end

function project!(cache,a::BlockHyperReduction,b::ArrayBlock)
  coeff,hypred = cache
  for i in eachindex(a)
    if a.touched[i]
      project!((coeff[i],blocks(hypred)[i]),a[i],b[i])
    end
  end
  return hypred
end

for (T,S) in zip((:HyperReduction,:BlockHyperReduction,:AffineContribution),
                 (:AbstractParamArray,:ArrayBlock,:ArrayContribution))
  @eval begin
    function project(a::$T,b::$S)
      coeff = allocate_coefficient(a,b)
      b̂ = allocate_hyper_reduction(a,b)
      project!((coeff,b̂),a,b)
    end
  end
end
