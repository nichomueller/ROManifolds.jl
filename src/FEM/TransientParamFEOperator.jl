"""
A parametric version of the `Gridap` `TransientFEOperator`
"""

abstract type TransientParamFEOperator{T<:ODEParamOperatorType} <: TransientFEOperator{T} end

function FESpaces.get_algebraic_operator(feop::TransientParamFEOperator)
  ODEParamOpFromTFEOp(feop)
end

function ODEs.allocate_tfeopcache(
  feop::TransientParamFEOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}})

  nothing
end

function ODEs.update_tfeopcache!(
  tfeopcache,
  feop::TransientParamFEOperator,
  r::TransientParamRealization)

  tfeopcache
end

realization(op::TransientParamFEOperator;kwargs...) = @abstractmethod
get_induced_norm(op::TransientParamFEOperator) = @abstractmethod

function assemble_norm_matrix(op::TransientParamFEOperator)
  @abstractmethod
end

get_coupling(op::TransientParamFEOperator) = @abstractmethod

function assemble_coupling_matrix(op::TransientParamFEOperator)
  @abstractmethod
end

get_linear_operator(op::TransientParamFEOperator) = @abstractmethod
get_nonlinear_operator(op::TransientParamFEOperator) = @abstractmethod

struct TransientParamFEOpFromWeakForm <: TransientParamFEOperator{NonlinearODE}
  res::Function
  jacs::Tuple{Vararg{Function}}
  induced_norm::Function
  assem::Assembler
  tpspace::TransientParamSpace
  trial::FESpace
  test::FESpace
  order::Integer
end

function TransientParamFEOperator(
  res::Function,jacs::Tuple{Vararg{Function}},induced_norm::Function,tpspace,trial,test)

  order = length(jacs) - 1
  assem = SparseMatrixAssembler(trial,test)
  TransientParamFEOpFromWeakForm(
    res,jacs,induced_norm,assem,tpspace,trial,test,order)
end

function TransientParamFEOperator(
  res::Function,jac::Function,induced_norm::Function,tpspace,trial,test)

  TransientParamFEOperator(res,(jac,),induced_norm,tpspace,trial,test)
end

function TransientParamFEOperator(
  res::Function,jac::Function,jac_t::Function,induced_norm::Function,tpspace,trial,test)

  TransientParamFEOperator(res,(jac,jac_t),induced_norm,tpspace,trial,test)
end

function TransientParamFEOperator(res::Function,induced_norm::Function,tpspace,trial,test;order::Integer=1)
  function jac_0(μ,t,u,du,v)
    function res_0(y)
      u0 = TransientCellField(y,u.derivatives)
      res(μ,t,u0,v)
    end
    jacobian(res_0,u.cellfield)
  end
  jacs = (jac_0,)

  for k in 1:order
    function jac_k(μ,t,u,duk,v)
      function res_k(y)
        derivatives = (u.derivatives[1:k-1]...,y,u.derivatives[k+1:end]...)
        uk = TransientCellField(u.cellfield,derivatives)
        res(μ,t,uk,v)
      end
      jacobian(res_k,u.derivatives[k])
    end
    jacs = (jacs...,jac_k)
  end

  TransientParamFEOperator(res,jacs,induced_norm,tpspace,trial,test)
end

FESpaces.get_test(op::TransientParamFEOpFromWeakForm) = op.test
FESpaces.get_trial(op::TransientParamFEOpFromWeakForm) = op.trial
Polynomials.get_order(op::TransientParamFEOpFromWeakForm) = op.order
ODEs.get_res(tfeop::TransientParamFEOpFromWeakForm) = tfeop.res
ODEs.get_jacs(tfeop::TransientParamFEOpFromWeakForm) = tfeop.jacs
ODEs.get_assembler(tfeop::TransientParamFEOpFromWeakForm) = tfeop.assem
realization(op::TransientParamFEOpFromWeakForm;kwargs...) = realization(op.tpspace;kwargs...)
get_induced_norm(op::TransientParamFEOpFromWeakForm) = op.induced_norm

function assemble_norm_matrix(op::TransientParamFEOpFromWeakForm)
  test = get_test(op)
  trial = evaluate(get_trial(op),nothing)
  inorm = get_induced_norm(op)
  assemble_matrix(inorm,trial,test)
end

function ODEs.get_assembler(feop::TransientParamFEOpFromWeakForm,r::TransientParamRealization)
  get_param_assembler(get_assembler(feop),r)
end

struct TransientParamLinearFEOpFromWeakForm <: TransientParamFEOperator{LinearODE}
  forms::Tuple{Vararg{Function}}
  res::Function
  jacs::Tuple{Vararg{Function}}
  constant_forms::Tuple{Vararg{Bool}}
  induced_norm::Function
  tpspace::TransientParamSpace
  assem::Assembler
  trial::FESpace
  test::FESpace
  order::Integer
end

function TransientParamLinearFEOperator(
  forms::Tuple{Vararg{Function}},res::Function,jacs::Tuple{Vararg{Function}},
  induced_norm::Function,tpspace,trial,test;
  constant_forms::Tuple{Vararg{Bool}}=ntuple(_ -> false, length(forms)))

  order = length(jacs) - 1
  assem = SparseMatrixAssembler(trial,test)
  TransientParamLinearFEOpFromWeakForm(
    forms,res,jacs,constant_forms,induced_norm,tpspace,assem,trial,test,order)
end

function TransientParamLinearFEOperator(
  forms::Tuple{Vararg{Function}},res::Function,induced_norm::Function,tpspace,trial,test;kwargs...)

  jacs = ntuple(k -> ((t,u,duk,v) -> forms[k](t,duk,v)),order+1)
  TransientParamLinearFEOperator(forms,res,jacs,induced_norm,tpspace,trial,test;kwargs...)
end

function TransientParamLinearFEOperator(
  mass::Function,res::Function,induced_norm::Function,tpspace,trial,test;
  constant_forms::NTuple{1,Bool}=(false,))

  TransientParamLinearFEOperator((mass,),res,induced_norm,tpspace,trial,test;constant_forms)
end

function TransientParamLinearFEOperator(
  stiffness::Function,mass::Function,res::Function,induced_norm::Function,tpspace,trial,test;
  constant_forms::NTuple{2,Bool}=(false,false))

  TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,tpspace,trial,test;constant_forms)
end

FESpaces.get_test(op::TransientParamLinearFEOpFromWeakForm) = op.test
FESpaces.get_trial(op::TransientParamLinearFEOpFromWeakForm) = op.trial
Polynomials.get_order(op::TransientParamLinearFEOpFromWeakForm) = op.order
ODEs.get_res(tfeop::TransientParamLinearFEOpFromWeakForm) = tfeop.res
ODEs.get_jacs(tfeop::TransientParamLinearFEOpFromWeakForm) = tfeop.jacs
ODEs.get_assembler(tfeop::TransientParamLinearFEOpFromWeakForm) = tfeop.assem
realization(op::TransientParamLinearFEOpFromWeakForm;kwargs...) = realization(op.tpspace;kwargs...)
get_induced_norm(op::TransientParamLinearFEOpFromWeakForm) = op.induced_norm

function assemble_norm_matrix(op::TransientParamLinearFEOpFromWeakForm)
  test = get_test(op)
  trial = evaluate(get_trial(op),nothing)
  inorm = get_induced_norm(op)
  assemble_matrix(inorm,trial,test)
end

function ODEs.get_assembler(feop::TransientParamLinearFEOpFromWeakForm,r::TransientParamRealization)
  get_param_assembler(get_assembler(feop),r)
end

function TransientFETools.test_transient_fe_operator(op::TransientParamFEOperator,uh,μt)
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
  jacobians!(J,op,μt,xh,(1.0,1.0),cache)
  @test isa(J,ParamMatrix)
  cache = update_cache!(cache,op,μt)
  true
end
