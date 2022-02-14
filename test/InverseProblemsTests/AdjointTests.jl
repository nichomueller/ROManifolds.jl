using Gridap
using Gridap.FESpaces

# Manufactured solution
q̂(x) = 1.0 + x[1] + x[2]
û(x) = x[1] + 3*x[2]
σ̂(x) = q̂(x)*∇(û)(x)
f(x) = -(∇⋅σ̂)(x)

# Polynomial prior
q(p,x) = 1.0 + p[1] + p[2]*x[1] + p[3]*x[2]
dqdp(p,x) = ∇(k->q(k,x))(p)

# Target params
const p̂ = VectorValue(0,1,1)

function main(;cells=(10,10))

  # Geometry
  domain = (0,1,0,1); cells = (10,10)
  model = CartesianDiscreteModel(domain,cells)

  #FESpaces
  order_u = 1
  reffe_u = ReferenceFE(lagrangian,Float64,order_u)
  V = TestFESpace(model,reffe_u,dirichlet_tags="boundary")
  U = TrialFESpace(V,û)
  order_q = 1
  reffe_q = ReferenceFE(lagrangian,Float64,order_q)
  reffe_dqdp = ReferenceFE(lagrangian,typeof(p̂),order_q)
  Q = FESpace(model,reffe_q)

  # Integration
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*order_u)

  # Objective function.
  # I.e. the l2 error wrt the "target" u.
  # It returns a cell-wise value.
  # The actual objective is computed as sum(f(u))
  j(u) = ∫( abs2(u-û) )dΩ
  J = LossFunction(∫( abs2(u-û) )dΩ, U)

  # Conductivity-dependent weak form
  a(q,u,v) =  ∫( ∇(v)⋅(q*∇(u)) )dΩ
  l(v) = ∫( v*f )dΩ
  res(q,u,v) = a(q,u,v) - l(v)

  function uh_to_j(uh)
    sum(j(uh))
  end

  function uh_to_j_rrule(uh)
    function uh_to_j_pullback(dj)
      dfdu = ∇(j)(uh)
      b = assemble_vector(dfdu,V)
      dj*b
    end
    uh_to_j(uh), uh_to_j_pullback
  end

  function qh_to_uh(qh)
    op = AffineFEOperator((u,v)->a(qh,u,v),l,U,V)
    uh = solve(op)
    uh
  end

  function qh_to_uh_rrule(qh)
    # Direct problem
    op = AffineFEOperator((u,v)->a(qh,u,v),l,U,V)
    uh = solve(op)
    function qh_to_uh_pullback(duh)
      # Adjoint solve (assuming A self-adjoint)
      A = get_matrix(op)
      λh = FEFunction(V,A\duh)
      # Sensitivity wrt qh
      dadqh = ∇(q->a(q,uh,λh))(qh)
      vals = -1*assemble_vector(dadqh,Q)
      FEFunction(Q,vals)
    end
    uh, qh_to_uh_pullback
  end

  function p_to_qh(p)
    qh = interpolate(x->q(p,x),Q)
  end

  function p_to_qh_rrule(p)
    function p_to_qh_pullback(dqh)
      cell_dqh = get_cell_dof_values(dqh)
      cf = CellField(x->dqdp(p,x),Ω)
      x = get_cell_points(get_cell_dof_basis(Q))
      cell_dqhdp = cf(x)
      cell_dq = lazy_map(cell_dqh,cell_dqhdp) do dqh, dqhdp
        m = zero(p̂)
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

  function p_to_j(p)
    qh = p_to_qh(p)
    uh = qh_to_uh(qh)
    uh_to_j(uh)
  end

  function j_and_djdp(p)
    qh, qh_pullback = p_to_qh_rrule(p)
    uh, uh_pullback = qh_to_uh_rrule(qh)
    jp, j_pullback = uh_to_j_rrule(uh)
    dj = 1.0
    duh = j_pullback(dj)
    dqh = uh_pullback(duh)
    dp = qh_pullback(dqh)
    jp, dp
  end

  @show p_to_j(p̂)
  @show j_and_djdp(p̂)

  #writevtk(Ω,"results",cellfields=["û"=>û,"q̂"=>q̂])

end



qu_to_j

q_to_j + u_to_j * q_to_u

u_to_j_pullback(j̄)

q_to_u_pullback(ū)

dj=1.0 # identity operator

_, ∂j_q = q_to_j_pullback(dj)

_, dj_u = u_to_j_pullback(dj)

_, du_q = q_to_u_pullback(dj_u)

 = ∂j_q + dj_u
