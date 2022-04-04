module InverseProblems

using Gridap
using Gridap.FESpaces
# using Gridap.CellData
using ChainRulesCore

import ChainRulesCore.rrule

export FEStateMap
export LossFunction

# A function that is defined in terms of a set of parameters,
# e.g., an AbstractVector. We transform it into a FEFunction.
struct ParamsToFEFunction{P}
  fun::Function{P}
  Q::FESpace
end

ParamsToFEFunction(f::Function{T},Q::FESpace) = ParamsToFEFunction{T}(f,Q)

function ChainRulesCore.rrule(p_to_qh::ParamsToFEFunction{P},p)
  p_to_qh(p) = interpolate(x->p_to_qh.fun(p,x),Q)
  function p_to_qh_pullback(dqh)
    dqdp(p,x) = ∇(k->p_to_qh.fun(k,x))(p)
    cell_dqh = get_cell_dof_values(dqh)
    Ω = get_triangulation(p_to_qh.Q)
    cf = CellField(x->dqdp(p,x),Ω)
    x = get_cell_points(get_cell_dof_basis(Q))
    cell_dqhdp = cf(x)
    cell_dq = lazy_map(cell_dqh,cell_dqhdp) do dqh, dqhdp
      m = zero(P)
      for i in 1:length(dqh)
        m += dqhdp[i]*dqh[i]
      end
      m
    end
    dp = sum(cell_dq)
    dp
  end
  p_to_qh(p), p_to_qh_pullback
end

# Struct that represents the FE solution of a parameterised PDE.
# It is a map from the parameter space T to a FESpace
struct FEStateMap{P,U <: TrialFESpace, V <: FESpace}
  form::Function
  param_sp::P # params (CellData)
  test::U
  trial::V
  # assem::Assembler
end

# Add traits for Affine, Adjoint, etc. + other constructors
# Provide solve() functionality (linear and nonlinear solver def'on)
# Save Assembler in cache for efficiency (nonlinear problems, etc)
# Define promotion of different types to CellData (if not already done in Gridap)

function ChainRulesCore.rrule(uh::FEStateMap{T},qh::T) where T <: FEFunction
  # function qh_to_uh_rrule(qh)
  # Direct problem
  a = uh.form
  U = uh.test
  V = uh.trial
  Q = uh.param_sp
  op = FEOperator((u,v)->a(qh,u,v),U,V)
  uh = solve(op)
  function qh_to_uh_pullback(duh)
    A = get_matrix(op)
    λh = FEFunction(V,A'\duh)
    # Sensitivity wrt qh
    dadqh = ∇(q->a(q,uh,λh))(qh)
    vals = -1*assemble_vector(dadqh,Q)
    FEFunction(Q,vals)
    # We can probably define a method for these two lines
    # instead of dispatching with T
  end
  uh, qh_to_uh_pullback
end

struct LossFunction{P,U}
  loss::Function
  param_sp::P
  state_sp::U
  # assem::Assembler
end

# Here, we can define different constructors, depending on T
# E.g., P can be a FEFunction or a Real or an AbstractVector ...
# E.g., U can be a FEFunction (unconstrained) or a FEStateMap{P} (for PDE-constrained)

# This chain rule computes the derivative of a function with
# respect to a FEFunction. Conceptually, it derives the
# functional with respect to the free DOFs in the FEFunction,
# and returns a vector with one entry for each DOF.
function ChainRulesCore.rrule(j::LossFunction{FEFunction,FEStateMap}, qh::FEFunction)

  uh = j.state_sp # Space of PDE solutions over Q
  Q = j.param_sp

  qh_to_uh, qh_to_uh_pullback = ChainRulesCore.rrule(uh,qh)
  qh_to_j = sum(j.loss(qh,uh))

  function uh_to_j_pullback(dj)
    dj_u = ∇(u -> j.loss(qh,u))(qh_to_uh)
    b = assemble_vector(djdu,U)
    dj*b
  end

  function qh_to_j_pullback(dj)
    ∂j_q= ∇(q -> j.loss(q,qh_to_uh))(qh)
    b = assemble_vector(∂j_q,Q)
    jq = dj*b
    ju = uh_to_j_pullback(dj)
    juq = qh_to_uh_pullback(ju)
    return jq + juq
  end

  qh_to_j,  qh_to_j_pullback

end

end # module
