"""
    abstract type ParamFEOperator{T<:UnEvalOperatorType} <: FEOperator end

Parametric extension of a [`FEOperator`](@ref) in [`Gridap`](@ref). Compared to
a standard FEOperator, there are the following novelties:

- a ParamSpace is provided, so that parametric realizations can be extracted
  directly from the ParamFEOperator
- an AbstractIndexMap is provided, so that a nonstandard indexing strategy can
  take place when dealing with FEFunctions
- a function representing a norm matrix is provided, so that errors in the
  desired norm can be automatically computed

Subtypes:

- [`ParamFEOpFromWeakForm`](@ref)

"""
abstract type ParamFEOperator{T<:UnEvalOperatorType} <: FEOperator end

function FESpaces.get_test(feop::ParamFEOperator)
  @abstractmethod
end

function FESpaces.get_trial(feop::ParamFEOperator)
  @abstractmethod
end

function get_param_space(feop::ParamFEOperator)
  @abstractmethod
end

function FESpaces.get_algebraic_operator(op::ParamFEOperator)
  ParamOpFromFEOp(op)
end

function allocate_feopcache(op::ParamFEOperator,r::Realization,u::AbstractVector)
  nothing
end

function update_feopcache!(feop_cache,op::ParamFEOperator,u::AbstractVector)
  feop_cache
end

ParamDataStructures.realization(op::ParamFEOperator;kwargs...) = realization(get_param_space(op);kwargs...)

function ParamFESpaces.get_param_assembler(op::ParamFEOperator,r::AbstractRealization)
  get_param_assembler(get_assembler(op),r)
end

IndexMaps.get_index_map(op::ParamFEOperator) = @abstractmethod
IndexMaps.get_vector_index_map(op::ParamFEOperator) = get_vector_index_map(get_index_map(op))
IndexMaps.get_matrix_index_map(op::ParamFEOperator) = get_matrix_index_map(get_index_map(op))

function FESpaces.assemble_matrix(op::ParamFEOperator,form::Function)
  test = get_test(op)
  trial = evaluate(get_trial(op),nothing)
  _assemble_matrix(form,trial,test)
end

function _assemble_matrix(f,U::FESpace,V::FESpace)
  assemble_matrix(f,U,V)
end

function _assemble_matrix(f,U::TProductFESpace,V::TProductFESpace)
  a = SparseMatrixAssembler(U,V)
  v = get_tp_fe_basis(V)
  u = get_tp_trial_fe_basis(U)
  assemble_matrix(a,collect_cell_matrix(U,V,f(u,v)))
end

function _assemble_matrix(f,U::TrialFESpace{<:TProductFESpace},V::TProductFESpace)
  _assemble_matrix(f,U.space,V)
end

function _assemble_matrix(f,U::MultiFieldFESpace,V::MultiFieldFESpace)
  if all(isa.(V.spaces,TProductFESpace))
    a = TProductBlockSparseMatrixAssembler(U,V)
    v = get_tp_fe_basis(V)
    u = get_tp_trial_fe_basis(U)
    assemble_matrix(a,collect_cell_matrix(U,V,f(u,v)))
  else
    assemble_matrix(f,U,V)
  end
end

"""
    struct ParamFEOpFromWeakForm{T} <: ParamFEOperator{T} end

Most standard instance of ParamFEOperator{T}

"""
struct ParamFEOpFromWeakForm{T} <: ParamFEOperator{T}
  res::Function
  jac::Function
  pspace::ParamSpace
  assem::Assembler
  index_map::FEOperatorIndexMap
  trial::FESpace
  test::FESpace
end

function ParamFEOperator(
  res::Function,jac::Function,pspace,trial,test)

  assem = SparseMatrixAssembler(trial,test)
  index_map = FEOperatorIndexMap(trial,test)
  ParamFEOpFromWeakForm{NonlinearParamEq}(res,jac,pspace,assem,index_map,trial,test)
end

function LinearParamFEOperator(
  res::Function,jac::Function,pspace,trial,test)

  jac′(μ,u,du,v,args...) = jac(μ,du,v,args...)
  assem = SparseMatrixAssembler(trial,test)
  index_map = FEOperatorIndexMap(trial,test)
  ParamFEOpFromWeakForm{LinearParamEq}(res,jac′,pspace,constant_jac,assem,index_map,trial,test)
end

FESpaces.get_test(op::ParamFEOpFromWeakForm) = op.test
FESpaces.get_trial(op::ParamFEOpFromWeakForm) = op.trial
get_param_space(op::ParamFEOpFromWeakForm) = op.pspace
ODEs.get_res(op::ParamFEOpFromWeakForm) = op.res
get_jac(op::ParamFEOpFromWeakForm) = op.jac
ODEs.get_assembler(op::ParamFEOpFromWeakForm) = op.assem
IndexMaps.get_index_map(op::ParamFEOpFromWeakForm) = op.index_map
