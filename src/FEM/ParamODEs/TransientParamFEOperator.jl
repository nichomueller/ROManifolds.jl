"""
    abstract type TransientParamFEOperator{T<:ODEParamOperatorType} <: TransientFEOperator{T} end

Parametric extension of a [`TransientFEOperator`](@ref) in [`Gridap`](@ref). Compared to
a standard TransientFEOperator, there are the following novelties:

- a TransientParamSpace is provided, so that transient parametric realizations
  can be extracted directly from the TransientParamFEOperator
- an AbstractIndexMap is provided, so that a nonstandard indexing strategy can
  take place when dealing with FEFunctions
- a function representing a norm matrix is provided, so that errors in the
  desired norm can be automatically computed

Subtypes:

- [`TransientParamFEOpFromWeakForm`](@ref)
- [`TransientParamLinearFEOpFromWeakForm`](@ref)

"""
abstract type TransientParamFEOperator{O<:ODEParamOperatorType,T<:TriangulationStyle} <: ParamFEOperator{O,T} end
const JointTransientParamFEOperator{O<:ODEParamOperatorType} = TransientParamFEOperator{O,JointTriangulation}
const SplitTransientParamFEOperator{O<:ODEParamOperatorType} = TransientParamFEOperator{O,SplitTriangulation}

function FESpaces.get_algebraic_operator(op::TransientParamFEOperator)
  ODEParamOpFromTFEOp(op)
end

function ParamSteady.allocate_feopcache(
  op::TransientParamFEOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}})

  nothing
end

function ParamSteady.update_feopcache!(
  feop_cache,
  op::TransientParamFEOperator,
  us::Tuple{Vararg{AbstractVector}})

  feop_cache
end

ODEs.get_res(op::TransientParamFEOperator) = @abstractmethod

ODEs.get_jacs(op::TransientParamFEOperator) = @abstractmethod

function Polynomials.get_order(feop::TransientParamFEOperator)
  @abstractmethod
end

"""
    struct TransientParamFEOpFromWeakForm <: TransientParamFEOperator{NonlinearParamODE} end

Most standard instance of TransientParamFEOperator, when the transient problem is
nonlinear

"""
struct TransientParamFEOpFromWeakForm{T} <: TransientParamFEOperator{NonlinearParamODE,T}
  res::Function
  jacs::Tuple{Vararg{Function}}
  tpspace::TransientParamSpace
  assem::Assembler
  index_map::FEOperatorIndexMap
  trial::FESpace
  test::FESpace
  order::Integer
end

const JointTransientParamFEOpFromWeakForm = TransientParamFEOpFromWeakForm{JointTriangulation}

function TransientParamFEOperator(
  res::Function,jacs::Tuple{Vararg{Function}},tpspace,trial,test)

  order = length(jacs) - 1
  assem = SparseMatrixAssembler(trial,test)
  index_map = FEOperatorIndexMap(trial,test)
  TransientParamFEOpFromWeakForm{JointTriangulation}(
    res,jacs,tpspace,assem,index_map,trial,test,order)
end

const SplitTransientParamFEOpFromWeakForm = TransientParamFEOpFromWeakForm{SplitTriangulation}

function TransientParamFEOperator(
  res::Function,jacs::Tuple{Vararg{Function}},tpspace,trial,test,trian_res,trian_jacs...)

  order = length(jacs) - 1
  res′,jacs′ = _set_triangulations(res,jacs,test,trial,trian_res,trian_jacs)
  assem = SparseMatrixAssembler(trial,test)
  index_map = FEOperatorIndexMap(trial,test,trian_res,trian_jacs)
  TransientParamFEOpFromWeakForm{SplitTriangulation}(
    res′,jacs′,tpspace,assem,index_map,trial,test,order)
end

function TransientParamFEOperator(
  res::Function,jac::Function,tpspace,trial,test,args...)

  TransientParamFEOperator(res,(jac,),tpspace,trial,test,args...)
end

function TransientParamFEOperator(
  res::Function,jac::Function,jac_t::Function,tpspace,trial,test,args...)

  TransientParamFEOperator(res,(jac,jac_t),tpspace,trial,test,args...)
end

function TransientParamFEOperator(
  res::Function,tpspace,trial,test,args...;order::Integer=1)

  function jac_0(μ,t,u,du,v,args...)
    function res_0(y)
      u0 = TransientCellField(y,u.derivatives)
      res(μ,t,u0,v,args...)
    end
    jacobian(res_0,u.cellfield)
  end
  jacs = (jac_0,)

  for k in 1:order
    function jac_k(μ,t,u,duk,v,args...)
      function res_k(y)
        derivatives = (u.derivatives[1:k-1]...,y,u.derivatives[k+1:end]...)
        uk = TransientCellField(u.cellfield,derivatives)
        res(μ,t,uk,v,args...)
      end
      jacobian(res_k,u.derivatives[k])
    end
    jacs = (jacs...,jac_k)
  end

  TransientParamFEOperator(res,jacs,tpspace,trial,test,args...)
end

FESpaces.get_test(op::TransientParamFEOpFromWeakForm) = op.test
FESpaces.get_trial(op::TransientParamFEOpFromWeakForm) = op.trial
ParamSteady.get_param_space(op::TransientParamFEOpFromWeakForm) = op.tpspace
Polynomials.get_order(op::TransientParamFEOpFromWeakForm) = op.order
ODEs.get_res(op::TransientParamFEOpFromWeakForm) = op.res
ODEs.get_jacs(op::TransientParamFEOpFromWeakForm) = op.jacs
ODEs.get_assembler(op::TransientParamFEOpFromWeakForm) = op.assem
IndexMaps.get_index_map(op::TransientParamFEOpFromWeakForm) = op.index_map

"""
    struct TransientParamLinearFEOpFromWeakForm <: TransientParamFEOperator{LinearParamODE} end

Most standard instance of TransientParamFEOperator, when the transient problem is
linear

"""
struct TransientParamLinearFEOpFromWeakForm{T} <: TransientParamFEOperator{LinearParamODE,T}
  res::Function
  jacs::Tuple{Vararg{Function}}
  constant_forms::Tuple{Vararg{Bool}}
  tpspace::TransientParamSpace
  assem::Assembler
  index_map::FEOperatorIndexMap
  trial::FESpace
  test::FESpace
  order::Integer
end

const JointTransientParamLinearFEOpFromWeakForm = TransientParamLinearFEOpFromWeakForm{JointTriangulation}

function TransientParamLinearFEOperator(
  forms::Tuple{Vararg{Function}},res::Function,tpspace,trial,test;
  constant_forms::Tuple{Vararg{Bool}}=ntuple(_ -> false,length(forms)))

  order = length(forms)-1
  jacs = ntuple(k -> ((μ,t,u,duk,v) -> forms[k](μ,t,duk,v)),length(forms))
  assem = SparseMatrixAssembler(trial,test)
  index_map = FEOperatorIndexMap(trial,test)
  TransientParamLinearFEOpFromWeakForm{JointTriangulation}(
    res,jacs,constant_forms,tpspace,assem,index_map,trial,test,order)
end

const SplitTransientParamLinearFEOpFromWeakForm = TransientParamLinearFEOpFromWeakForm{SplitTriangulation}

function TransientParamLinearFEOperator(
  forms::Tuple{Vararg{Function}},res::Function,tpspace,trial,test,trian_res,trian_jacs...;
  constant_forms::Tuple{Vararg{Bool}}=ntuple(_ -> false,length(forms)))

  order = length(forms) - 1
  jacs = ntuple(k -> ((μ,t,u,duk,v,args...) -> forms[k](μ,t,duk,v,args...)),length(forms))
  res′,jacs′ = _set_triangulations(res,jacs,test,trial,trian_res,trian_jacs)
  assem = SparseMatrixAssembler(trial,test)
  index_map = FEOperatorIndexMap(trial,test,trian_res,trian_jacs)
  TransientParamLinearFEOpFromWeakForm{SplitTriangulation}(
    res′,jacs′,constant_forms,tpspace,assem,index_map,trial,test,order)
end

function TransientParamLinearFEOperator(
  mass::Function,res::Function,tpspace,trial,test,args...;kwargs...)

  TransientParamLinearFEOperator((mass,),res,tpspace,trial,test;kwargs)
end

function TransientParamLinearFEOperator(
  stiffness::Function,mass::Function,res::Function,tpspace,trial,test,args...;kwargs...)

  TransientParamLinearFEOperator((stiffness,mass),res,tpspace,trial,test,args...;kwargs...)
end

function TransientParamLinearFEOperator(
  stiffness::Function,damping::Function,mass::Function,res::Function,
  tpspace,trial,test,args...;kwargs...)

  TransientParamLinearFEOperator((stiffness,damping,mass),res,tpspace,trial,test,args...;kwargs...)
end

FESpaces.get_test(op::TransientParamLinearFEOpFromWeakForm) = op.test
FESpaces.get_trial(op::TransientParamLinearFEOpFromWeakForm) = op.trial
ParamSteady.get_param_space(op::TransientParamLinearFEOpFromWeakForm) = op.tpspace
Polynomials.get_order(op::TransientParamLinearFEOpFromWeakForm) = op.order
ODEs.get_res(op::TransientParamLinearFEOpFromWeakForm) = op.res
ODEs.get_jacs(op::TransientParamLinearFEOpFromWeakForm) = op.jacs
ODEs.get_assembler(op::TransientParamLinearFEOpFromWeakForm) = op.assem
ODEs.is_form_constant(op::TransientParamLinearFEOpFromWeakForm,k::Integer) = op.constant_forms[k]
IndexMaps.get_index_map(op::TransientParamLinearFEOpFromWeakForm) = op.index_map

# triangulation utils

for (f,T) in zip((:(Utils.set_domains),:(Utils.change_domains)),(:JointTriangulation,:SplitTriangulation))
  @eval begin
    function $f(op::SplitTransientParamFEOpFromWeakForm,trian_res,trian_jacs)
      trian_res′ = order_triangulations(get_trian_res(op),trian_res)
      trian_jacs′ = map(order_triangulations,get_trian_jac(op),trian_jacs)
      res′,jacs′ = _set_triangulations(op.res,op.jacs,op.test,op.trial,trian_res′,trian_jacs′)
      index_map′ = $f(op.index_map,trian_res′,trian_jacs′)
      TransientParamFEOpFromWeakForm{$T}(
        res′,jacs′,op.tpspace,op.assem,index_map′,op.trial,op.test,op.order)
    end

    function $f(op::SplitTransientParamLinearFEOpFromWeakForm,trian_res,trian_jacs)
      trian_res′ = order_triangulations(get_trian_res(op),trian_res)
      trian_jacs′ = map(order_triangulations,get_trian_jac(op),trian_jacs)
      res′,jacs′ = _set_triangulations(op.res,op.jacs,op.test,op.trial,trian_res′,trian_jacs′)
      index_map′ = $f(op.index_map,trian_res′,trian_jacs′)
      TransientParamLinearFEOpFromWeakForm{$T}(
        res′,jacs′,op.constant_forms,op.tpspace,
        op.assem,index_map′,op.trial,op.test,op.order)
    end
  end
end

function _set_triangulation_jac(
  jac::Function,
  trian::Tuple{Vararg{Triangulation}},
  order)

  degree = 2*order
  meas = Measure.(trian,degree)
  newjac(μ,t,u,du,v,args...) = jac(μ,t,u,du,v,args...)
  newjac(μ,t,u,du,v) = newjac(μ,t,u,du,v,meas...)
  return newjac
end

function _set_triangulation_jacs(
  jacs::Tuple{Vararg{Function}},
  trians::Tuple{Vararg{Tuple{Vararg{Triangulation}}}},
  order)

  newjacs = ()
  for (jac,trian) in zip(jacs,trians)
    newjacs = (newjacs...,_set_triangulation_jac(jac,trian,order))
  end
  return newjacs
end

function _set_triangulation_form(
  res::Function,
  trian::Tuple{Vararg{Triangulation}},
  order)

  degree = 2*order
  meas = Measure.(trian,degree)
  newres(μ,t,u,v,args...) = res(μ,t,u,v,args...)
  newres(μ,t,u,v) = newres(μ,t,u,v,meas...)
  return newres
end

function _set_triangulations(
  res::Function,
  jacs::Tuple{Vararg{Function}},
  test::FESpace,
  trial::FESpace,
  trian_res::Tuple{Vararg{Triangulation}},
  trian_jacs::Tuple{Vararg{Tuple{Vararg{Triangulation}}}})

  polyn_order = get_polynomial_order(test)
  @check polyn_order == get_polynomial_order(trial)
  res′ = _set_triangulation_form(res,trian_res,polyn_order)
  jacs′ = _set_triangulation_jacs(jacs,trian_jacs,polyn_order)
  return res′,jacs′
end
