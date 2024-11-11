"""
    abstract type ParamFEOperator{O<:UnEvalOperatorType,T<:TriangulationStyle} <: FEOperator end

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
abstract type ParamFEOperator{O<:UnEvalOperatorType,T<:TriangulationStyle} <: FEOperator end
const JointParamFEOperator{O<:UnEvalOperatorType} = ParamFEOperator{O,JointTriangulation}
const SplitParamFEOperator{O<:UnEvalOperatorType} = ParamFEOperator{O,SplitTriangulation}

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
    struct ParamFEOpFromWeakForm{O,T} <: ParamFEOperator{O,T} end

Most standard instance of ParamFEOperator{O,T}

"""
struct ParamFEOpFromWeakForm{O,T} <: ParamFEOperator{O,T}
  res::Function
  jac::Function
  pspace::ParamSpace
  assem::Assembler
  index_map::FEOperatorIndexMap
  trial::FESpace
  test::FESpace
end

const JointParamFEOpFromWeakForm{O} = ParamFEOpFromWeakForm{O,JointTriangulation}

function ParamFEOperator(res::Function,jac::Function,pspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  index_map = FEOperatorIndexMap(trial,test)
  ParamFEOpFromWeakForm{NonlinearParamEq,JointTriangulation}(
    res,jac,pspace,assem,index_map,trial,test)
end

function LinearParamFEOperator(res::Function,jac::Function,pspace,trial,test)
  jac′(μ,u,du,v) = jac(μ,du,v)
  assem = SparseMatrixAssembler(trial,test)
  index_map = FEOperatorIndexMap(trial,test)
  ParamFEOpFromWeakForm{LinearParamEq,JointTriangulation}(
    res,jac′,pspace,assem,index_map,trial,test)
end

const SplitParamFEOpFromWeakForm{O} = ParamFEOpFromWeakForm{O,SplitTriangulation}

function ParamFEOperator(res::Function,jac::Function,pspace,trial,test,trian_res,trian_jac)
  res′,jac′ = _set_triangulations(res,jac,test,trial,trian_res,trian_jac)
  assem = SparseMatrixAssembler(trial,test)
  index_map = FEOperatorIndexMap(trial,test,trian_res,trian_jac)
  ParamFEOpFromWeakForm{NonlinearParamEq,SplitTriangulation}(
    res′,jac′,pspace,assem,index_map,trial,test)
end

function LinearParamFEOperator(res::Function,jac::Function,pspace,trial,test,trian_res,trian_jac)
  jac′(μ,u,du,v,args...) = jac(μ,du,v,args...)
  res′,jac′′ = _set_triangulations(res,jac′,test,trial,trian_res,trian_jac)
  assem = SparseMatrixAssembler(trial,test)
  index_map = FEOperatorIndexMap(trial,test,trian_res,trian_jac)
  ParamFEOpFromWeakForm{LinearParamEq,SplitTriangulation}(
    res′,jac′′,pspace,assem,index_map,trial,test)
end

FESpaces.get_test(op::ParamFEOpFromWeakForm) = op.test
FESpaces.get_trial(op::ParamFEOpFromWeakForm) = op.trial
get_param_space(op::ParamFEOpFromWeakForm) = op.pspace
ODEs.get_res(op::ParamFEOpFromWeakForm) = op.res
get_jac(op::ParamFEOpFromWeakForm) = op.jac
ODEs.get_assembler(op::ParamFEOpFromWeakForm) = op.assem
IndexMaps.get_index_map(op::ParamFEOpFromWeakForm) = op.index_map

# triangulation utils

get_trian_res(op::ParamFEOperator) = get_domains(get_vector_index_map(op))
get_trian_jac(op::ParamFEOperator) = get_domains(get_matrix_index_map(op))

function Utils.set_domains(op::JointParamFEOperator,args...)
  op
end

function Utils.set_domains(op::SplitParamFEOperator)
  set_domains(op,get_trian_res(op),get_trian_jac(op))
end

function Utils.change_domains(op::JointParamFEOperator)
  @notimplemented "The triangulations of this operator are fixed"
end

function Utils.change_domains(op::ParamFEOperator)
  @notimplemented "Need to provide triangulations for this function"
end

for (f,T) in zip((:(Utils.set_domains),:(Utils.change_domains)),(:JointTriangulation,:SplitTriangulation))
  @eval begin
    function $f(op::SplitParamFEOpFromWeakForm{O},trian_res,trian_jac) where O
      trian_res′ = order_triangulations(get_trian_res(op),trian_res)
      trian_jac′ = order_triangulations(get_trian_jac(op),trian_jac)
      res′,jac′ = _set_triangulations(op.res,op.jac,op.trial,op.test,trian_res′,trian_jac′)
      index_map′ = $f(op.index_map,op.test,op.trial,trian_res′,trian_jac′)
      ParamFEOpFromWeakForm{O,$T}(
        res′,jac′,op.pspace,op.assem,index_map′,op.trial,op.test)
    end
  end
end

function _set_triangulation_jac(
  jac::Function,
  trian::Tuple{Vararg{Triangulation}},
  order)

  degree = 2*order
  meas = Measure.(trian,degree)
  jac′(μ,u,du,v,args...) = jac(μ,u,du,v,args...)
  jac′(μ,u,du,v) = jac′(μ,u,du,v,meas...)
  return jac′
end

function _set_triangulation_res(
  res::Function,
  trian::Tuple{Vararg{Triangulation}},
  order)

  degree = 2*order
  meas = Measure.(trian,degree)
  res′(μ,u,v,args...) = res(μ,u,v,args...)
  res′(μ,u,v) = res′(μ,u,v,meas...)
  return res′
end

function _set_triangulations(
  res::Function,
  jac::Function,
  test::FESpace,
  trial::FESpace,
  trian_res::Tuple{Vararg{Triangulation}},
  trian_jac::Tuple{Vararg{Triangulation}})

  polyn_order = get_polynomial_order(test)
  @check polyn_order == get_polynomial_order(trial)
  res′ = _set_triangulation_res(res,trian_res,polyn_order)
  jac′ = _set_triangulation_jac(jac,trian_jac,polyn_order)
  return res′,jac′
end
