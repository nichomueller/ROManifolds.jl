function steady_poisson()
  uh,μ = fe_snapshots(ptype,solver,op,fepath,run_fem,1)
  opA = ParamVarOperator(a,afe,PS,U,V;id=:A)
  opF = AffineParamVarOperator(f,ffe,PS,V;id=:F)
  opH = ParamVarOperator(h,hfe,PS,V;id=:H)

  A,LA = assemble_matrix_and_lifting(opA)
  F = assemble_vector(opF)
  H = assemble_vector(opH)
  μ1 = μ[1]
  isapprox(A(μ1)*uh.snap,F(μ1)+H(μ1)-LA(μ1))
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
