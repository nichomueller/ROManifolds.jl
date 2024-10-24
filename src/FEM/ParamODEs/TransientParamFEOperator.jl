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
- [`TransientParamFEOperatorWithTrian`](@ref)

"""
abstract type TransientParamFEOperator{T<:ODEParamOperatorType} <: ParamFEOperator{T} end

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
struct TransientParamFEOpFromWeakForm <: TransientParamFEOperator{NonlinearParamODE}
  res::Function
  jacs::Tuple{Vararg{Function}}
  tpspace::TransientParamSpace
  assem::Assembler
  index_map::FEOperatorIndexMap
  trial::FESpace
  test::FESpace
  order::Integer
end

function TransientParamFEOperator(
  res::Function,jacs::Tuple{Vararg{Function}},tpspace,trial,test,args...)

  order = length(jacs) - 1
  assem = SparseMatrixAssembler(trial,test)
  index_map = FEOperatorIndexMap(trial,test)
  TransientParamFEOpFromWeakForm(
    res,jacs,tpspace,assem,index_map,trial,test,order,args...)
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
struct TransientParamLinearFEOpFromWeakForm <: TransientParamFEOperator{LinearParamODE}
  forms::Tuple{Vararg{Function}}
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

function TransientParamLinearFEOperator(
  forms::Tuple{Vararg{Function}},res::Function,tpspace,trial,test,
  args...;constant_forms::Tuple{Vararg{Bool}}=ntuple(_ -> false,length(forms)))

  order = length(forms)-1
  jacs = ntuple(k -> ((μ,t,u,duk,v,args...) -> forms[k](μ,t,duk,v,args...)),length(forms))
  assem = SparseMatrixAssembler(trial,test)
  index_map = FEOperatorIndexMap(trial,test)
  TransientParamLinearFEOpFromWeakForm(
    forms,res,jacs,constant_forms,tpspace,assem,index_map,trial,test,order,args...)
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

function test_transient_fe_operator(op::TransientParamFEOperator,uh,μt)
  odeop = get_algebraic_operator(op)
  @test isa(odeop,ODEParamOperator)
  cache = allocate_cache(op)
  V = get_test(op)
  @test isa(V,FESpace)
  U = get_trial(op)
  U0 = U(μt)
  @test isa(U0,TrialParamFESpace)
  r = allocate_residual(op,μt,uh,cache)
  @test isa(r,ParamVector)
  xh = TransientCellField(uh,(uh,))
  residual!(r,op,μt,xh,cache)
  @test isa(r,ParamVector)
  J = allocate_jacobian(op,μt,uh,cache)
  @test isa(J,ParamMatrix)
  jacobian!(J,op,μt,xh,1,1.0,cache)
  @test isa(J,ParamMatrix)
  jacobian!(J,op,μt,xh,2,1.0,cache)
  @test isa(J,ParamMatrix)
  cache = update_cache!(cache,op,μt)
  true
end
