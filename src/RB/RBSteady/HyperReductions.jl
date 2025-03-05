function empirical_interpolation!(cache,A::AbstractMatrix)
  I,res = cache
  m,n = size(A)
  resize!(res,m)
  resize!(I,n)
  @views I[1] = argmax(abs.(A[:,1]))
  if n > 1
    @inbounds for i = 2:n
      @views Bi = A[:,1:i-1]
      Ci = A[I[1:i-1],1:i-1]
      Di = A[I[1:i-1],i]
      @views res = A[:,i] - Bi*(Ci \ Di)
      I[i] = argmax(abs.(res))
    end
  end
  Ai = view(A,I,:)
  return I,Ai
end

function eim_cache(A::AbstractMatrix)
  m,n = size(A)
  res = zeros(eltype(A),m)
  I = zeros(Int,n)
  return I,res
end

function empirical_interpolation(A::AbstractArray)
  cache = eim_cache(A)
  I,AI = empirical_interpolation!(cache,A)
  return I,AI
end

function empirical_interpolation(A::ParamSparseMatrix)
  I,AI = empirical_interpolation(A.data)
  R′,C′ = recast_split_indices(I,param_getindex(A,1))
  return (R′,C′),AI
end

"""
    get_dofs_to_cells(
      cell_dof_ids::AbstractArray{<:AbstractArray},
      dofs::AbstractVector
      ) -> AbstractVector

Returns the list of cells containing the dof ids `dofs`
"""
function get_dofs_to_cells(
  cell_dof_ids::AbstractArray{<:AbstractArray},
  dofs::AbstractVector)

  cells = Int32[]
  cache = array_cache(cell_dof_ids)
  for cell = eachindex(cell_dof_ids)
    celldofs = getindex!(cache,cell_dof_ids,cell)
    if !isempty(intersect(dofs,celldofs))
      append!(cells,cell)
    end
  end
  return cells
end

function reduced_cells(
  test::FESpace,
  trian::Triangulation,
  rows::AbstractVector)

  cell_dof_ids = get_cell_dof_ids(test,trian)
  cells = get_dofs_to_cells(cell_dof_ids,rows)
  return cells
end

function reduced_cells(
  trial::FESpace,
  test::FESpace,
  trian::Triangulation,
  rows::AbstractVector,
  cols::AbstractVector)

  cell_dof_ids_trial = get_cell_dof_ids(trial,trian)
  cell_dof_ids_test = get_cell_dof_ids(test,trian)
  cells_trial = get_dofs_to_cells(cell_dof_ids_trial,cols)
  cells_test = get_dofs_to_cells(cell_dof_ids_test,cols)
  cells = union(cells_trial,cells_test)
  return cells
end

"""
    abstract type IntegrationDomain end

Type representing the set of interpolation rows of a `Projection` subjected
to a EIM approximation with `empirical_interpolation`.
Subtypes:
- [`VectorDomain`](@ref)
- [`TransientIntegrationDomain`](@ref)
"""
abstract type IntegrationDomain end

"""
    struct VectorDomain <: IntegrationDomain{Int,1}
      rows::Vector{Int}
      cells::Vector{Int32}
    end

Integration domain for a projection vector operator in a steady problem
"""
struct VectorDomain <: IntegrationDomain
  rows::Vector{Int}
  cells::Vector{Int32}
end

function vector_domain(args...)
  @abstractmethod
end

function vector_domain(
  test::FESpace,
  trian::Triangulation,
  rows::Vector{<:Number})

  cells = reduced_cells(test,trian,rows)
  VectorDomain(rows,cells)
end

"""
    struct MatrixDomain <: IntegrationDomain
      rows::Vector{Int}
      cols::Vector{Int}
      cells::Vector{Int32}
    end

Integration domain for a projection matrix operator in a steady problem
"""
struct MatrixDomain <: IntegrationDomain
  rows::Vector{Int}
  cols::Vector{Int}
  cells::Vector{Int32}
  function MatrixDomain(rows::Vector{Int},cols::Vector{Int},cells::Vector{Int32})
    @check length(rows) == length(cols)
    new(rows,cols,cells)
  end
end

function matrix_domain(args...)
  @abstractmethod
end

function matrix_domain(
  trial::FESpace,
  test::FESpace,
  trian::Triangulation,
  rows::Vector{<:Number},
  cols::Vector{<:Number})

  cells = reduced_cells(trial,test,trian,rows,cols)
  MatrixDomain(rows,cols,cells)
end

"""
    abstract type HyperReduction{
      A<:Reduction,
      B<:ReducedProjection,
      C<:AbstractIntegrationDomain
      } <: Projection end

Subtype of a [`Projection`](@ref) dedicated to the outputd of a hyper-reduction
(e.g. an empirical interpolation method (EIM)) procedure applied on residual/jacobians
of a differential problem. This procedure can be summarized in the following steps:

1. compute a snapshots tensor `T`
2. construct a `Projection` `Φ` by running the function `reduction` on `T`
3. find the EIM quantities `i`,`Φi` by running the function `empirical_interpolation`
  on `Φ`

The triplet (`Φ`, `Φi`,`i`) represents the minimum information needed to run the
online phase of the hyper-reduction. However, we recall that a RB method requires
the (Petrov-)Galerkin projection of residuals/Jacobianson a reduced subspace
built from solution snapshots, instead of providing the projection `Φ` we return
the reduced projection `Φrb`, where

- for residuals: `Φrb = test_basis' * Φ`
- for Jacobians: `Φrb = test_basis' * Φ * trial_basis`

The output of this operation is a ReducedProjection. Therefore, a HyperReduction
is completely characterized by the triplet `(Φrb,Φi,i)`.
Subtypes:
- [`EmptyHyperReduction`](@ref)
- [`MDEIM`](@ref)
"""
abstract type HyperReduction{A<:Reduction,B<:ReducedProjection,C<:AbstractIntegrationDomain} <: Projection end

HyperReduction(::Reduction,args...) = @abstractmethod

"""
    get_interpolation(a::HyperReduction) -> Factorization

For a [`HyperReduction`](@ref) `a` represented by the triplet `(Φrb,Φi,i)`,
returns `Φi`, usually stored as a Factorization
"""
get_interpolation(a::HyperReduction) = @abstractmethod

"""
    get_integration_domain(a::HyperReduction) -> AbstractIntegrationDomain

For a [`HyperReduction`](@ref) `a` represented by the triplet `(Φrb,Φi,i)`,
returns `i`
"""
get_integration_domain(a::HyperReduction) = @abstractmethod

num_reduced_dofs(a::HyperReduction) = num_reduced_dofs(get_basis(a))
num_reduced_dofs_left_projector(a::HyperReduction) = num_reduced_dofs_left_projector(get_basis(a))
num_reduced_dofs_right_projector(a::HyperReduction) = num_reduced_dofs_right_projector(get_basis(a))

function inv_project!(
  b̂::AbstractParamArray,
  coeff::AbstractParamArray,
  a::HyperReduction,
  b::AbstractParamArray)

  o = one(eltype2(b̂))
  interp = get_interpolation(a)
  ldiv!(coeff,interp,b)
  mul!(b̂,a,coeff,o,o)
  return b̂
end

"""
    struct EmptyHyperReduction{A,B} <: HyperReduction{A,B,IntegrationDomain}
      reduction::A
      basis::B
    end

Trivial hyper-reduction returned whenever the residual/Jacobian is zero
"""
struct EmptyHyperReduction{A,B} <: HyperReduction{A,B,IntegrationDomain}
  reduction::A
  basis::B
end

get_basis(a::EmptyHyperReduction) = a.basis
get_interpolation(a::EmptyHyperReduction) = @notimplemented
get_integration_domain(a::EmptyHyperReduction) = @notimplemented

function HyperReduction(
  red::Reduction,
  test::RBSpace)

  red = get_reduction(red)
  nrows = num_free_dofs(test)
  basis = ReducedProjection(zeros(nrows,1))
  return EmptyHyperReduction(red,basis)
end

function HyperReduction(
  red::Reduction,
  trial::RBSpace,
  test::RBSpace)

  red = get_reduction(red)
  nrows = num_free_dofs(test)
  ncols = num_free_dofs(trial)
  basis = ReducedProjection(zeros(nrows,1,ncols))
  return EmptyHyperReduction(red,basis)
end

"""
    struct MDEIM{A,B,C} <: HyperReduction{A,B,C}
      reduction::A
      basis::B
      interpolation::Factorization
      domain::C
    end

[`HyperReduction`](@ref) returned by a matrix-based empirical interpolation method
"""
struct MDEIM{A,B,C} <: HyperReduction{A,B,C}
  reduction::A
  basis::B
  interpolation::Factorization
  domain::C
end

get_basis(a::MDEIM) = a.basis
get_interpolation(a::MDEIM) = a.interpolation
get_integration_domain(a::MDEIM) = a.domain

function HyperReduction(
  red::AbstractMDEIMReduction,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace)

  red = get_reduction(red)
  basis = projection(red,s)
  proj_basis = project(test,basis)
  indices,interp = empirical_interpolation(basis)
  factor = lu(interp)
  domain = vector_domain(test,trian,indices)
  return MDEIM(red,proj_basis,factor,domain)
end

function HyperReduction(
  red::AbstractMDEIMReduction,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace)

  red = get_reduction(red)
  basis = projection(red,s)
  proj_basis = project(test,basis,trial)
  indices,interp = empirical_interpolation(basis)
  factor = lu(interp)
  domain = matrix_domain(trial,test,trian,indices...)
  return MDEIM(red,proj_basis,factor,domain)
end

"""
    reduced_triangulation(trian::Triangulation,a::HyperReduction)

Returns the triangulation view of `trian` on the integration cells contained in `a`
"""
function reduced_triangulation(trian::Triangulation,a::HyperReduction)
  i = get_integration_domain(a)
  cells = get_cells(i)
  red_trian = view(trian,cells)
  return red_trian
end

function allocate_coefficient(a::HyperReduction,r::AbstractRealization)
  n = num_reduced_dofs(a)
  np = num_params(r)
  coeffvec = zeros(n)
  coeff = global_parameterize(coeffvec,np)
  return coeff
end

function allocate_hyper_reduction(
  a::HyperReduction{<:Reduction,<:ReducedVecProjection},
  r::AbstractRealization)

  nrows = num_reduced_dofs_left_projector(a)
  np = num_params(r)
  b = zeros(nrows)
  hypred = global_parameterize(b,np)
  fill!(hypred,zero(eltype(hypred)))
  return hypred
end

function allocate_hyper_reduction(
  a::HyperReduction{<:Reduction,<:ReducedMatProjection},
  r::AbstractRealization)

  nrows = num_reduced_dofs_left_projector(a)
  ncols = num_reduced_dofs_right_projector(a)
  np = num_params(r)
  M = zeros(nrows,ncols)
  hypred = global_parameterize(M,np)
  fill!(hypred,zero(eltype(hypred)))
  return hypred
end

function Utils.Contribution(v::Tuple{Vararg{HyperReduction}},t::Tuple{Vararg{Triangulation}})
  AffineContribution(v,t)
end

"""
    struct AffineContribution{A,V,K} <: Contribution

Contribution whose `values` assume one of the following types:

- [`HyperReduction`](@ref) for single field problems
- [`BlockProjection`](@ref) for multi field problems
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

function allocate_coefficient(a::AffineContribution,r::AbstractRealization)
  contribution(get_domains(a)) do trian
    allocate_coefficient(a[trian],r)
  end
end

function allocate_hyper_reduction(a::AffineContribution,r::AbstractRealization)
  allocate_hyper_reduction(first(get_contributions(a)),r)
end

"""
"""
function allocate_hypred_cache(a::AffineContribution,r::AbstractRealization)
  fecache = allocate_coefficient(a,r)
  coeffs = allocate_coefficient(a,r)
  hypred = allocate_hyper_reduction(a,r)
  return HRParamArray(fecache,coeffs,hypred)
end

function inv_project!(
  hypred::AbstractParamArray,
  coeff::ArrayContribution,
  a::AffineContribution,
  b::ArrayContribution)

  @check length(coeff) == length(a) == length(b)
  fill!(hypred,zero(eltype(hypred)))
  for (aval,bval,cval) in zip(get_contributions(a),get_contributions(b),get_contributions(coeff))
    inv_project!(hypred,cval,aval,bval)
  end
  return hypred
end

function reduced_form(
  red::Reduction,
  args...)

  HyperReduction(red,args...)
end

function reduced_form(
  red::Reduction,
  s::Snapshots,
  trian::Triangulation,
  args...)

  hyper_red = HyperReduction(red,s,trian,args...)
  red_trian = reduced_triangulation(trian,hyper_red)
  return hyper_red,red_trian
end

"""
    reduced_residual(
      solver::RBSolver,
      op::ParamOperator,
      red_test::RBSpace,
      s::AbstractSnapshots
      ) -> AffineContribution

Reduces the residual contained in `op` via hyper-reduction. This function
first builds the residual snapshots, which are then reduced according to the strategy
`residual_reduction` specified in the reduced solver `solver`
"""
function reduced_residual(
  solver::RBSolver,
  op::ParamOperator,
  red_test::RBSpace,
  s::AbstractSnapshots)

  res = residual_snapshots(solver,op,s)
  res_red = get_residual_reduction(solver)
  reduced_residual(res_red,red_test,res)
end

function reduced_residual(red::Reduction,test::RBSpace,c::ArrayContribution)
  t = @timed begin
    a,trians = map(get_domains(c),get_contributions(c)) do trian,values
      reduced_form(red,values,trian,test)
    end |> tuple_of_arrays
  end
  println(CostTracker(t,name="Residual hyper-reduction"))
  return Contribution(a,trians)
end

"""
    reduced_jacobian(
      solver::RBSolver,
      op::ParamOperator,
      red_trial::RBSpace,
      red_test::RBSpace,
      s::AbstractSnapshots
      ) -> Union{AffineContribution,TupOfAffineContribution}

Reduces the Jacobian contained in `op` via hyper-reduction. This function
first builds the Jacobian snapshots, which are then reduced according to the strategy
`reduced_jacobian` specified in the reduced solver `solver`. In transient applications,
the output is a tuple of length equal to the number of Jacobians(i.e., equal to
the order of the ODE plus one)
"""
function reduced_jacobian(
  solver::RBSolver,
  op::ParamOperator,
  red_trial::RBSpace,
  red_test::RBSpace,
  s::AbstractSnapshots)

  jac = jacobian_snapshots(solver,op,s)
  jac_red = get_jacobian_reduction(solver)
  reduced_jacobian(jac_red,red_trial,red_test,jac)
end

function reduced_jacobian(red::Reduction,trial::RBSpace,test::RBSpace,c::ArrayContribution)
  t = @timed begin
    a,trians = map(get_domains(c),get_contributions(c)) do trian,values
      reduced_form(red,values,trian,trial,test)
    end |> tuple_of_arrays
  end
  println(CostTracker(t,name="Jacobian hyper-reduction"))
  return Contribution(a,trians)
end

"""
    reduced_weak_form(
      solver::RBSolver,
      op::ParamOperator,
      red_trial::RBSpace,
      red_test::RBSpace,
      s::AbstractSnapshots
      ) -> (AffineContribution,Union{AffineContribution,TupOfAffineContribution})

Reduces the residual/Jacobian contained in `op` via hyper-reduction. Check the
functions [`reduced_residual`](@ref) and [`reduced_jacobian`](@ref) for more details
"""
function reduced_weak_form(
  solver::RBSolver,
  op::ParamOperator,
  red_trial::RBSpace,
  red_test::RBSpace,
  s::AbstractSnapshots)

  red_jac = reduced_jacobian(solver,op,red_trial,red_test,s)
  red_res = reduced_residual(solver,op,red_test,s)
  return red_jac,red_res
end

# multi field interface

"""
    const BlockHyperReduction{A<:HyperReduction,N} = BlockProjection{A,N}
"""
const BlockHyperReduction{A<:HyperReduction,N} = BlockProjection{A,N}

function Utils.Contribution(
  v::Tuple{Vararg{BlockHyperReduction}},
  t::Tuple{Vararg{Triangulation}})

  AffineContribution(v,t)
end

for f in (:get_basis,:get_interpolation,:get_integration_domain)
  @eval begin
    function Arrays.return_cache(::typeof($f),a::HyperReduction)
      cache = $f(a)
      return cache
    end

    function Arrays.return_cache(::typeof($f),a::BlockHyperReduction)
      i = findfirst(a.touched)
      @notimplementedif isnothing(i)
      cache = return_cache($f,a[i])
      block_cache = Array{typeof(cache),ndims(a)}(undef,size(a))
      return block_cache
    end

    function $f(a::BlockHyperReduction)
      cache = return_cache($f,a)
      for i in eachindex(a)
        if a.touched[i]
          cache[i] = $f(a[i])
        end
      end
      return ArrayBlock(cache,a.touched)
    end
  end
end

function inv_project!(
  hypred::BlockParamArray,
  coeff::ArrayBlock,
  a::BlockHyperReduction,
  b::ArrayBlock)

  for i in eachindex(a)
    if a.touched[i]
      inv_project!(blocks(hypred)[i],coeff[i],a[i],b[i])
    end
  end
  return hypred
end

for (T,S) in zip((:AffineContribution,:BlockHyperReduction),(:ArrayContribution,:ArrayBlock))
  @eval begin
    function inv_project(a::$T,b::$S)
      @notimplemented "Must provide cache in advance"
    end

    function inv_project!(cache::HRParamArray,a::$T,b::$S)
      inv_project!(cache.hypred,cache.coeff,a,b)
    end
  end
end

function Arrays.return_cache(
  ::typeof(allocate_coefficient),
  a::HyperReduction,
  r::AbstractRealization)

  coeffvec = testvalue(Vector{Float64})
  global_parameterize(coeffvec,num_params(r))
end

function Arrays.return_cache(
  ::typeof(allocate_coefficient),
  a::BlockHyperReduction,
  r::AbstractRealization)

  i = findfirst(a.touched)
  @notimplementedif isnothing(i)
  coeff = return_cache(allocate_coefficient,a[i],r)
  block_coeff = Array{typeof(coeff),ndims(a)}(undef,size(a))
  return block_coeff
end

function allocate_coefficient(a::BlockHyperReduction,r::AbstractRealization)
  coeff = return_cache(allocate_coefficient,a,r)
  for i in eachindex(a)
    if a.touched[i]
      coeff[i] = allocate_coefficient(a[i],r)
    end
  end
  return ArrayBlock(coeff,a.touched)
end

function Arrays.return_cache(
  ::typeof(allocate_hyper_reduction),
  a::HyperReduction{<:Reduction,<:ReducedVecProjection},
  r::AbstractRealization)

  hypvec = testvalue(Vector{Float64})
  global_parameterize(hypvec,num_params(r))
end

function Arrays.return_cache(
  ::typeof(allocate_hyper_reduction),
  a::HyperReduction{<:Reduction,<:ReducedMatProjection},
  r::AbstractRealization)

  hypvec = testvalue(Matrix{Float64})
  global_parameterize(hypvec,num_params(r))
end

function Arrays.return_cache(
  ::typeof(allocate_hyper_reduction),
  a::BlockHyperReduction,
  r::AbstractRealization)

  i = findfirst(a.touched)
  @notimplementedif isnothing(i)
  hypred = return_cache(allocate_hyper_reduction,a[i],r)
  block_hypred = Array{typeof(hypred),ndims(a)}(undef,size(a))
  return block_hypred
end

function allocate_hyper_reduction(a::BlockHyperReduction,r::AbstractRealization)
  hypred = return_cache(allocate_hyper_reduction,a,r)
  for i in eachindex(a)
    hypred[i] = allocate_hyper_reduction(a.array[i],r)
  end
  return mortar(hypred)
end

function reduced_triangulation(trian::Triangulation,b::BlockHyperReduction)
  red_trian = Triangulation[]
  for i in eachindex(b)
    if b.touched[i]
      push!(red_trian,reduced_triangulation(trian,b[i]))
    end
  end
  return red_trian
end

function reduced_form(
  red::Reduction,
  s::BlockSnapshots,
  trian::Triangulation,
  test::MultiFieldRBSpace)

  hyper_reds = Vector{HyperReduction}(undef,size(s))
  red_trians = Triangulation[]
  for i in eachindex(s)
    if s.touched[i]
      hyper_red,red_trian = reduced_form(red,s[i],trian,test[i])
      hyper_reds[i] = hyper_red
      push!(red_trians,red_trian)
    else
      hyper_reds[i] = reduced_form(red,test[i])
    end
  end

  hyper_red = BlockProjection(hyper_reds,s.touched)
  red_trian = Utils.merge_triangulations(red_trians)

  return hyper_red,red_trian
end

function reduced_form(
  red::Reduction,
  s::BlockSnapshots,
  trian::Triangulation,
  trial::MultiFieldRBSpace,
  test::MultiFieldRBSpace)

  hyper_reds = Matrix{HyperReduction}(undef,size(s))
  red_trians = Triangulation[]
  for (i,j) in Iterators.product(axes(s)...)
    if s.touched[i,j]
      hyper_red,red_trian = reduced_form(red,s[i,j],trian,trial[j],test[i])
      hyper_reds[i,j] = hyper_red
      push!(red_trians,red_trian)
    else
      hyper_reds[i,j] = reduced_form(red,trial[j],test[i])
    end
  end

  hyper_red = BlockProjection(hyper_reds,s.touched)
  red_trian = Utils.merge_triangulations(red_trians)

  return hyper_red,red_trian
end
