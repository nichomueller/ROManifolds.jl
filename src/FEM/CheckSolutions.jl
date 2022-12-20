function steady_poisson()
  uh,μ = fe_snapshots(ptype,solver,op,fepath,run_fem,1)
  opA = ParamVarOperator(a,afe,PS,U,V;id=:A)
  opF = ParamVarOperator(f,ffe,PS,V;id=:F)
  opH = ParamVarOperator(h,hfe,PS,V;id=:H)

  A,LA = assemble_matrix_and_lifting(opA)
  F = assemble_vector(opF)
  H = assemble_vector(opH)
  μ1 = μ[1]
  isapprox(A(μ1)*uh.snap,F(μ1)+H(μ1)-LA(μ1))
end

function unsteady_poisson()
  dt = get_dt(opA)
  θ = get_θ(opA)
  μ1 = μ[1]
  u1 = uh.snap[:,1]
  u2 = uh.snap[:,2]

  tθ1 = dt*θ
  A1,LA1 = assemble_matrix_and_lifting(opA,tθ1)
  M1,LM1 = assemble_matrix_and_lifting(opM,tθ1)
  F1 = assemble_vector(opF,tθ1)
  H1 = assemble_vector(opH,tθ1)
  isapprox(θ*(A1(μ1)+M1(μ1)/(dt*θ))*u1,F1(μ1)+H1(μ1)-LA1(μ1)-LM1(μ1))

  tθ2 = dt+dt*θ
  A2,LA2 = assemble_matrix_and_lifting(opA,tθ2)
  M2,LM2 = assemble_matrix_and_lifting(opM,tθ2)
  F2 = assemble_vector(opF,tθ2)
  H2 = assemble_vector(opH,tθ2)
  LHS2 = θ*(A2(μ1)+M2(μ1)/(dt*θ))
  RHS2 = F2(μ1)+H2(μ1)-LA2(μ1)-LM2(μ1)
  RHS1 = ((1-θ)*A2(μ1) - θ*M2(μ1)/(dt*θ))*u1
  isapprox(LHS2*u2,RHS2-RHS1)
end

function steady_stokes()
  u1,p1,μ1 = uh.snap[:,1],ph.snap[:,1],μ[1]
  Np = size(p1,1)
  xh = vcat(u1,p1)

  A,LA = assemble_matrix_and_lifting(opA)
  B,LB = assemble_matrix_and_lifting(opB)
  F = assemble_vector(opF)
  H = assemble_vector(opH)
  μ1 = μ[1]
  LHS = vcat(hcat(A(μ1),-B(μ1)'),(hcat(B(μ1),zeros(Np,Np))))
  RHS = vcat(F(μ1)+H(μ1)-LA(μ1),-LB(μ1))

  isapprox(LHS*xh,RHS)
end

function unsteady_poisson()
  dt = get_dt(opA)
  θ = get_θ(opA)
  μ1 = μ[1]
  Nu = size(uh.snap[:,1],1)
  Np = size(ph.snap[:,1],1)
  x1 = vcat(uh.snap[:,1],ph.snap[:,1])
  x2 = vcat(uh.snap[:,2],ph.snap[:,2])

  ALA(t) = assemble_matrix_and_lifting(opA,t)
  BLB(t) = assemble_matrix_and_lifting(opB,t)
  MLM(t) = assemble_matrix_and_lifting(opM,t)
  F(μ,t) = assemble_vector(opF,t)(μ)
  H(μ,t) = assemble_vector(opH,t)(μ)
  A(μ,t) = ALA(t)[1](μ)
  LA(μ,t) = ALA(t)[2](μ)
  B(μ,t) = BLB(t)[1](μ)
  LB(μ,t) = BLB(t)[2](μ)
  M(μ,t) = MLM(t)[1](μ)
  LM(μ,t) = MLM(t)[2](μ)

  LHS(μ,t) = vcat(hcat(A(μ,t)+M(μ,t)/(dt*θ),-B(μ,t)'),hcat(B(μ,t),zeros(Np,Np)))
  RHS(μ,t) = vcat(F(μ,t)+H(μ,t)-LA(μ,t)-LM(μ,t),-LB(μ,t))
  RHSadd(μ,t,u) = vcat(M(μ,t)*u/(dt*θ),zeros(Np,1))
  xθ(μ,t,u) = LHS(μ,t) \ (RHS(μ,t)+RHSadd(μ,t,u))
  x(μ,t,x) = (xθ(μ,t,x[1:Nu,:])-(1-θ)*x)/θ

  tθ1 = dt*θ
  tθ2 = dt+dt*θ
  isapprox(x(μ1,tθ1,zeros(Nu+Np,1)),x1)
  isapprox(x(μ1,tθ2,x1),x2)
end

function navier_steady_stokes()
  u1,p1,μ1 = uh.snap[:,1],ph.snap[:,1],μ[1]
  Np = size(p1,1)
  x1 = vcat(u1,p1)
  μ1 = μ[1]

  A,LA = assemble_matrix_and_lifting(opA)
  B,LB = assemble_matrix_and_lifting(opB)
  C,LC = assemble_matrix_and_lifting(opC)
  D = assemble_matrix(opD)
  F = assemble_vector(opF)
  H = assemble_vector(opH)

  LHS(u) = vcat(hcat(A(μ1)+C(u),-B(μ1)'),(hcat(B(μ1),zeros(Np,Np))))
  RHS(u) = vcat(F(μ1)+H(μ1)-LA(μ1)-LC(u),-LB(μ1))
  J(x) = vcat(hcat(A(μ1)+C(x[1])+D(x[1]),-B(μ1)'),(hcat(B(μ1),zeros(Np,Np))))
  Res(x,xh) = LHS(x[1])*xh-RHS(x[1])

  xnewt = newton(Res,J,X(μ1))
  isapprox(xnewt,x1)
end

function spaces_steady()
  PS = ParamSpace(ranges,sampling)
  μ = realization(PS)
  μ0 = Param(zeros(size(μ.μ)))

  ptype = ProblemType(true,false,false)
  reffe = Gridap.ReferenceFE(lagrangian,Float,degree)

  function g(x,p::Param)
    μ = get_μ(p)
    1. + sum(Point(μ[4:6]) .* x)
  end
  g(p::Param) = x->g(x,p)
  g0(x,p::Param) = 1.
  g0(p::Param) = x->g0(x,p)

  V = MyTests(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = MyTrials(V,g,ptype)
  U0 = MyTrials(V,g0,ptype)

  isapprox(U.trial(μ0).dirichlet_values,U0.trial(μ).dirichlet_values)

  ptypevec = ProblemType(true,true,false)
  reffevec = Gridap.ReferenceFE(lagrangian,VectorValue{3,Float},degree)

  function gvec(x,p::Param)
    μ = get_μ(p)
    1. .+ Point(μ[4:6]) .* x
  end
  gvec(p::Param) = x->gvec(x,p)
  g0vec(x,p::Param) = VectorValue(1.,1.,1.)
  g0vec(p::Param) = x->g0vec(x,p)

  Vvec = MyTests(model,reffevec;conformity=:H1,dirichlet_tags=["dirichlet"])
  Uvec = MyTrials(Vvec,gvec,ptypevec)
  U0vec = MyTrials(Vvec,g0vec,ptypevec)

  isapprox(Uvec.trial(μ0).dirichlet_values,U0vec.trial(μ).dirichlet_values)
end

function spaces_unsteady()
  PS = ParamSpace(ranges,sampling)
  μ = realization(PS)
  μ0 = Param(zeros(size(μ.μ)))

  ptype = ProblemType(false,false,false)
  reffe = Gridap.ReferenceFE(lagrangian,Float,degree)

  function g(x,p::Param,t::Real)
    μ = get_μ(p)
    1. + t*sum(Point(μ[4:6]) .* x)
  end
  g(p::Param,t::Real) = x->g(x,p,t)
  g0(x,p::Param,t::Real) = 1.
  g0(p::Param,t::Real) = x->g0(x,p,t)

  V = MyTests(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = MyTrials(V,g,ptype)
  U0 = MyTrials(V,g0,ptype)

  isapprox(U.trial(μ0,0.).dirichlet_values,U0.trial(μ,1.).dirichlet_values)

  ptypevec = ProblemType(false,true,false)
  reffevec = Gridap.ReferenceFE(lagrangian,VectorValue{3,Float},degree)

  function gvec(x,p::Param,t::Real)
    μ = get_μ(p)
    1. .+ t*Point(μ[4:6]) .* x
  end
  gvec(p::Param,t::Real) = x->gvec(x,p,t)
  g0vec(x,p::Param,t::Real) = VectorValue(1.,1.,1.)
  g0vec(p::Param,t::Real) = x->g0vec(x,p,t)

  Vvec = MyTests(model,reffevec;conformity=:H1,dirichlet_tags=["dirichlet"])
  Uvec = MyTrials(Vvec,gvec,ptypevec)
  U0vec = MyTrials(Vvec,g0vec,ptypevec)

  isapprox(Uvec.trial(μ0,0.).dirichlet_values,U0vec.trial(μ,1.).dirichlet_values)
end
