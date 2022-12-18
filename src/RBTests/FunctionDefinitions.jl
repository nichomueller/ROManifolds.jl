function poisson_functions(ptype::ProblemType,measures::ProblemMeasures)
  poisson_functions(issteady(ptype),measures)
end

function poisson_functions(::Val{true},measures::ProblemFixedMeasures)

  function a(x,p::Param)
    μ = get_μ(p)
    1. + μ[6] + 1. / μ[5] * exp(-norm(x-Point(μ[1:3]))^2 / μ[4])
  end
  a(p::Param) = x->a(x,p)
  function f(x,p::Param)
    1.
  end
  f(p::Param) = x->f(x,p)
  function h(x,p::Param)
    μ = get_μ(p)
    1. + μ[6] + 1. / μ[5] * exp(-norm(x-Point(μ[1:3]))^2 / μ[4])
  end
  h(p::Param) = x->h(x,p)
  function g(x,p::Param)
    μ = get_μ(p)
    (1. + cos(norm(Point(μ[1:3]).*x))) / norm(1. + cos(norm(Point(μ[1:3]).*x)))
  end
  g(p::Param) = x->g(x,p)

  afe(p::Param,dΩ,u,v) = ∫(a(p) * ∇(v) ⋅ ∇(u))dΩ
  ffe(p::Param,dΩ,v) = ∫(f(p) * v)dΩ
  hfe(p::Param,dΓn,v) = ∫(h(p) * v)dΓn

  dΩ,dΓn = get_dΩ(measures),get_dΓn(measures)
  afe(p::Param,u,v) = afe(p,dΩ,u,v)
  afe(p::Param) = (u,v) -> afe(p,u,v)
  ffe(p::Param,v) = ffe(p,dΩ,v)
  ffe(p::Param) = v -> ffe(p,v)
  hfe(p::Param,v) = hfe(p,dΓn,v)
  hfe(p::Param) = v -> hfe(p,v)

  lhs(p,u,v) = afe(p,u,v)
  rhs(p,v) = ffe(p,v) + hfe(p,v)

  a,afe,f,ffe,h,hfe,g,lhs,rhs
end

function poisson_functions(::Val{false},measures::ProblemFixedMeasures)

  function a(x,p::Param,t::Real)
    μ = get_μ(p)
    1. + μ[6] + 1. / μ[5] * exp(-sin(t)*norm(x-Point(μ[1:3]))^2 / μ[4])
  end
  a(p::Param,t::Real) = x->a(x,p,t)
  a(p::Param) = t->a(p,t)
  function m(x,p::Param,t::Real)
    1.
  end
  m(p::Param,t::Real) = x->m(x,p,t)
  m(p::Param) = t->m(p,t)
  function f(x,p::Param,t::Real)
    μ = get_μ(p)
    1.
  end
  f(p::Param,t::Real) = x->f(x,p,t)
  f(p::Param) = t->f(p,t)
  function h(x,p::Param,t::Real)
    μ = get_μ(p)
    1. + sin(t)*sum(Point(μ[3:5]) .* x)
  end
  h(p::Param,t::Real) = x->h(x,p,t)
  h(p::Param) = t->h(p,t)
  function g(x,p::Param,t::Real)
    μ = get_μ(p)
    g_xpt = 1. + sin(t)*sum(Point(μ[4:6]) .* x)
    g_xpt / norm(g_xpt)
  end
  g(p::Param,t::Real) = x->g(x,p,t)
  g(p::Param) = t->g(p,t)

  afe(p::Param,t::Real,dΩ,u,v) = ∫(a(p,t) * ∇(v) ⋅ ∇(u))dΩ
  mfe(p,t,dΩ,u,v) = ∫(v ⋅ u)dΩ
  ffe(p::Param,t::Real,dΩ,v) = ∫(f(p,t) * v)dΩ
  hfe(p::Param,t::Real,dΓn,v) = ∫(h(p,t) * v)dΓn

  dΩ,dΓn = get_dΩ(measures),get_dΓn(measures)
  afe(p::Param,t::Real,u,v) = afe(p,t,dΩ,u,v)
  afe(p::Param,t::Real) = (u,v) -> afe(p,t,u,v)
  mfe(p::Param,t::Real,u,v) = mfe(p,t,dΩ,u,v)
  mfe(p::Param,t::Real) = (u,v) -> mfe(p,t,u,v)
  ffe(p::Param,t::Real,v) = ffe(p,t,dΩ,v)
  ffe(p::Param,t::Real) = v -> ffe(p,t,v)
  hfe(p::Param,t::Real,v) = hfe(p,t,dΓn,v)
  hfe(p::Param,t::Real) = v -> hfe(p,t,v)

  lhs(p,t,u,v) = afe(p,t,u,v)
  rhs(p,t,v) = ffe(p,t,v) + hfe(p,t,v)

  a,afe,m,mfe,f,ffe,h,hfe,g,lhs,rhs
end

function stokes_functions(ptype::ProblemType,measures::ProblemMeasures)
  stokes_functions(issteady(ptype),measures)
end

function stokes_functions(::Val{true},measures::ProblemFixedMeasures)

  function a(x,p::Param)
    μ = get_μ(p)
    5*μ[1] + abs(sin(norm(x.*Point(μ[1:3])))*μ[4])
  end
  a(p::Param) = x->a(x,p)
  b(x,p::Param) = 1.
  b(p::Param) = x->b(x,p)
  f(x,p::Param) = VectorValue(0.,0.,0.)
  f(p::Param) = x->f(x,p)
  h(x,p::Param) = VectorValue(0.,0.,0.)
  h(p::Param) = x->h(x,p)
  function g(x,p::Param)
    μ = get_μ(p)
    R = 0.5
    #xc = x-Point(R,R,R)
    #VectorValue(sum((xc.*Point(μ[1:3]))^2),0.,0.)/norm(μ[1:3])^2
    xc = x-Point(0,R,R)
    VectorValue(sum((xc.*Point(μ[1:3]))^2),0.,0.)*(x[1]==0)/norm(μ[1:3])^2
  end
  g(p::Param) = x->g(x,p)

  afe(p::Param,dΩ,u,v) = ∫(a(p) * ∇(v) ⊙ ∇(u))dΩ
  bfe(p::Param,dΩ,u,q) = ∫(b(p) * q * (∇⋅(u)))dΩ
  ffe(p::Param,dΩ,v) = ∫(f(p) ⋅ v)dΩ
  hfe(p::Param,dΓn,v) = ∫(h(p) ⋅ v)dΓn

  dΩ,dΓn = get_dΩ(measures),get_dΓn(measures)
  afe(p::Param,u,v) = afe(p,dΩ,u,v)
  afe(p::Param) = (u,v) -> afe(p,u,v)
  bfe(p::Param,u,v) = bfe(p,dΩ,u,v)
  bfe(p::Param) = (u,v) -> bfe(p,u,v)
  ffe(p::Param,v) = ffe(p,dΩ,v)
  ffe(p::Param) = v -> ffe(p,v)
  hfe(p::Param,v) = hfe(p,dΓn,v)
  hfe(p::Param) = v -> hfe(p,v)

  rhs(μ,(v,q)) = ffe(μ,v) + hfe(μ,v)
  lhs(μ,(u,p),(v,q)) = afe(μ,u,v) - bfe(μ,v,p) - bfe(μ,u,q)

  a,afe,b,bfe,f,ffe,h,hfe,g,lhs,rhs
end

function stokes_functions(::Val{false},measures::ProblemFixedMeasures)

  function a(x,p::Param,t::Real)
    μ = get_μ(p)
    5*μ[1] + abs(sin(t*norm(x.*Point(μ[1:3])))*μ[4])
  end
  a(p::Param,t::Real) = x->a(x,p,t)
  a(p::Param) = t->a(p,t)
  m(x,p::Param,t::Real) = 1.
  m(p::Param,t::Real) = x->m(x,p,t)
  m(p::Param) = t->m(p,t)
  b(x,p::Param,t::Real) = 1.
  b(p::Param,t::Real) = x->b(x,p,t)
  b(p::Param) = t->b(p,t)
  f(x,p::Param,t::Real) = VectorValue(0.,0.,0.)
  f(p::Param,t::Real) = x->f(x,p,t)
  h(x,p::Param,t::Real) = VectorValue(0.,0.,0.)
  h(p::Param,t::Real) = x->h(x,p,t)
  function g(x,p::Param,t::Real)
    μ = get_μ(p)
    R = 0.5
    xc = x-Point(0,R,R)
    (1. +eps()-cos(t))*VectorValue(sum((xc.*Point(μ[1:3]))^2),0.,0.)*(x[1]==0)/norm(μ[1:3])^2
  end
  g(p::Param,t::Real) = x->g(x,p,t)

  afe(p::Param,t::Real,dΩ,u,v) = ∫(a(p,t) * ∇(v) ⊙ ∇(u))dΩ
  mfe(p::Param,t::Real,dΩ,u,v) = ∫(v ⋅ u)dΩ
  bfe(p::Param,t::Real,dΩ,u,q) = ∫(b(p,t) * q * (∇⋅(u)))dΩ
  bTfe(p::Param,t::Real,dΩ,u,q) = ∫(b(p,t) * (∇⋅(q)) * u)dΩ
  ffe(p::Param,t::Real,dΩ,v) = ∫(f(p,t) ⋅ v)dΩ
  hfe(p::Param,t::Real,dΓn,v) = ∫(h(p,t) ⋅ v)dΓn

  dΩ,dΓn = get_dΩ(measures),get_dΓn(measures)
  afe(p::Param,t::Real,u,v) = afe(p,t,dΩ,u,v)
  afe(p::Param,t::Real) = (u,v) -> afe(p,t,u,v)
  mfe(p::Param,t::Real,u,v) = mfe(p,t,dΩ,u,v)
  mfe(p::Param,t::Real) = (u,v) -> mfe(p,t,u,v)
  bfe(p::Param,t::Real,u,v) = bfe(p,t,dΩ,u,v)
  bfe(p::Param,t::Real) = (u,v) -> bfe(p,t,u,v)
  bTfe(p::Param,t::Real,u,v) = bTfe(p,t,dΩ,u,v)
  bTfe(p::Param,t::Real) = (u,v) -> bTfe(p,t,u,v)
  ffe(p::Param,t::Real,v) = ffe(p,t,dΩ,v)
  ffe(p::Param,t::Real) = v -> ffe(p,t,v)
  hfe(p::Param,t::Real,v) = hfe(p,t,dΓn,v)
  hfe(p::Param,t::Real) = v -> hfe(p,t,v)

  rhs(μ,t,(v,q)) = ffe(μ,t,v) + hfe(μ,t,v)
  mfe_gridap(μ,t,(u,p),(v,q)) = mfe(μ,t,u,v)
  lhs(μ,t,(u,p),(v,q)) = afe(μ,t,u,v) - bfe(μ,t,v,p) - bfe(μ,t,u,q)

  a,afe,m,mfe,mfe_gridap,b,bfe,bTfe,f,ffe,h,hfe,g,lhs,rhs
end

function navier_stokes_functions(ptype::ProblemType,measures::ProblemMeasures)
  navier_stokes_functions(issteady(ptype),measures)
end

function navier_stokes_functions(::Val{true},measures::ProblemFixedMeasures)

  function a(x,p::Param)
    μ = get_μ(p)
    1. + μ[6] + 1. / μ[5] * exp(-norm(x-Point(μ[1:3]))^2 / μ[4])
  end
  a(μ::Param) = x->a(μ,x)
  b(μ::Param,x) = 1.
  b(μ::Param) = x->b(μ,x)
  function f(x,p::Param)
    μ = get_μ(p)
    1. + Point(μ[4:6]) .* x
  end
  f(μ::Param) = x->f(μ,x)
  function h(x,p::Param)
    μ = get_μ(p)
    1. + Point(μ[4:6]) .* x
  end
  h(μ::Param) = x->h(μ,x)
  function g(x,p::Param)
    μ = get_μ(p)
    1. + Point(μ[4:6]) .* x
  end
  g(μ::Param) = x->g(μ,x)

  afe(μ,u,v) = ∫(a(μ) * ∇(v) ⊙ ∇(u))dΩ
  bfe(μ,u,q) = ∫(b(μ) * q * (∇⋅(u)))dΩ
  cfe(z,u,v) = ∫(c(realization(P))*v⊙(∇(u)'⋅z))dΩ
  cfe(z) = (u,v) -> cfe(z,u,v)
  dfe(z,u,v) = ∫(d(realization(P))*v⊙(∇(z)'⋅u))dΩ
  dfe(z) = (u,v) -> dfe(z,u,v)
  ffe(μ,v) = ∫(f(μ) ⋅ v)dΩ
  hfe(μ,v) = ∫(h(μ) ⋅ v)dΓn

  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  cgridap(u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
  dcgridap(u,du,v) = ∫(v⊙(dconv∘(du,∇(du),u,∇(u))))dΩ

  rhs(μ,(v,q)) = ffe(μ,v) + hfe(μ,v)
  lhs(μ,(u,p),(v,q)) = afe(μ,u,v) - bfe(μ,v,p) - bfe(μ,u,q)

  res(μ,(u,p),(v,q)) = lhs(μ,(u,p),(v,q)) + c(u,v) - rhs(μ,(v,q))
  jac(μ,(u,p),(du,dp),(v,q)) = lhs(μ,(du,dp),(v,q)) + dc(u,du,v)

  a,b,f,h,g,afe,bfe,cfe,dfe,ffe,hfe,res,jac
end

function navier_stokes_functions(
  dΩ::Measure,
  dΓn::Measure,
  ::Val{false})

  function a(x,p::Param,t::Real)
    μ = get_μ(p)
    1. + μ[6] + 1. / μ[5] * exp(-sin(t)*norm(x-Point(μ[1:3]))^2 / μ[4])
  end
  a(μ::Param,t::Real) = x->a(x,μ,t)
  a(μ::Param) = t->a(μ,t)

  b(x,μ::Param,t::Real) = 1.
  b(μ::Param,t::Real) = x->b(x,μ,t)
  b(μ::Param) = t->b(μ,t)

  c(x,μ::Param,t::Real) = 1.
  c(μ::Param,t::Real) = x->c(x,μ,t)
  c(μ::Param) = t->c(μ,t)

  d(x,μ::Param,t::Real) = 1.
  d(μ::Param,t::Real) = x->d(x,μ,t)
  d(μ::Param) = t->d(μ,t)

  function f(x,p::Param,t::Real)
    μ = get_μ(p)
    1. + sin(t)*Point(μ[4:6]).*x
  end
  f(μ::Param,t::Real) = x->f(x,μ,t)
  f(μ::Param) = t->f(μ,t)

  function h(x,p::Param,t::Real)
    μ = get_μ(p)
    1. + sin(t)*Point(μ[4:6]).*x
  end
  h(μ::Param,t::Real) = x->h(x,μ,t)
  h(μ::Param) = t->h(μ,t)

  function g(x,p::Param,t::Real)
    μ = get_μ(p)
    1. + sin(t)*Point(μ[4:6]).*x
  end
  g(μ::Param,t::Real) = x->g(x,μ,t)
  g(μ::Param) = t->g(μ,t)

  m(x,μ::Param,t::Real) = 1.
  m(μ::Param,t::Real) = x->m(x,μ,t)
  m(μ::Param) = t->m(μ,t)

  mfe(μ,t,u,v) = ∫(v⋅u)dΩ
  afe(μ,t,u,v) = ∫(a(μ,t)*∇(v) ⊙ ∇(u))dΩ
  bfe(μ,t,u,q) = ∫(b(μ,t)*q*(∇⋅(u)))dΩ
  cfe(μ,t,z,u,v) = ∫(c(μ,t)*v⊙(∇(u)'⋅z))dΩ
  cfe(μ,t,z) = (u,v) -> cfe(μ,t,z,u,v)
  cfe(z) = cfe(realization(P),0.,z)
  dfe(μ,t,z,u,v) = ∫(d(μ,t)*v⊙(∇(z)'⋅u))dΩ
  dfe(μ,t,z) = (u,v) -> dfe(μ,t,z,u,v)
  dfe(z) = dfe(realization(P),0.,z)
  ffe(μ,t,v) = ∫(f(μ,t)⋅v)dΩ
  hfe(μ,t,v) = ∫(h(μ,t)⋅v)dΓn
  #FOR SOME REASON, BETTER CONVERGENCE WITH THIS DEF OF NONLINEAR TERM
  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  cgridap(u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
  dcgridap(u,du,v) = ∫(v⊙(dconv∘(du,∇(du),u,∇(u))))dΩ

  rhs(μ,t,(v,q)) = ffe(μ,t,v) + hfe(μ,t,v)
  lhs(μ,t,(u,p),(v,q)) = afe(μ,t,u,v) - bfe(μ,t,v,p) - bfe(μ,t,u,q)

  res(μ,t,(u,p),(v,q)) = mfe(μ,t,(∂t(u),∂t(p)),(v,q)) + lhs(μ,t,(u,p),(v,q)) + cgridap(u,v) - rhs(μ,t,(v,q))
  jac(μ,t,(u,p),(du,dp),(v,q)) = lhs(μ,t,(du,dp),(v,q)) + dcgridap(u,du,v)
  jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = mfe(μ,t,(dut,dpt),(v,q))

  a,b,c,d,f,h,m,g,afe,bfe,cfe,dfe,ffe,hfe,mfe,res,jac,jac_t
end
