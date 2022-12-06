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
    μ = get_μ(p)
    1. #+ sum(Point(μ[2:4]) .* x)
  end
  f(p::Param) = x->f(x,p)
  function h(x,p::Param)
    μ = get_μ(p)
    1. + sum(Point(μ[3:5]) .* x)
  end
  h(p::Param) = x->h(x,p)
  function g(x,p::Param)
    μ = get_μ(p)
    1. + sum(Point(μ[4:6]) .* x)
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
    1. + μ[6] + 1. / μ[5] * exp(norm(x-Point(μ[1:3]))^2 / μ[4])
  end
  a(p::Param,t::Real) = x->a(x,p,t)
  a(p::Param) = t->a(p,t)
  function f(x,p::Param,t::Real)
    μ = get_μ(p)
    1.
  end
  f(p::Param,t::Real) = x->f(x,p,t)
  f(p::Param) = t->f(p,t)
  function h(x,p::Param,t::Real)
    μ = get_μ(p)
    1. + sum(Point(μ[3:5]) .* x)
  end
  h(p::Param,t::Real) = x->h(x,p,t)
  h(p::Param) = t->h(p,t)
  function g(x,p::Param,t::Real)
    μ = get_μ(p)
    1. + sum(Point(μ[4:6]) .* x)
  end
  g(p::Param,t::Real) = x->g(x,p,t)
  g(p::Param) = t->g(p,t)

  afe(p::Param,t::Real,dΩ,u,v) = ∫(a(p,t) * ∇(v) ⋅ ∇(u))dΩ
  ffe(p::Param,t::Real,dΩ,v) = ∫(f(p,t) * v)dΩ
  hfe(p::Param,t::Real,dΓn,v) = ∫(h(p,t) * v)dΓn

  dΩ,dΓn = get_dΩ(measures),get_dΓn(measures)
  afe(p::Param,t::Real,u,v) = afe(p,t,dΩ,u,v)
  afe(p::Param,t::Real) = (u,v) -> afe(p,t,u,v)
  ffe(p::Param,t::Real,v) = ffe(p,t,dΩ,v)
  ffe(p::Param,t::Real) = v -> ffe(p,t,v)
  hfe(p::Param,t::Real,v) = hfe(p,t,dΓn,v)
  hfe(p::Param,t::Real) = v -> hfe(p,v)

  lhs(p,t,u,v) = afe(p,t,u,v)
  rhs(p,t,v) = ffe(p,t,v) + hfe(p,t,v)
  mfe(p,t,u,v) = ∫(0. * v ⋅ u)dΩ

  a,afe,f,ffe,h,hfe,g,mfe,lhs,rhs
end

function navier_stokes_functions(dΩ::Measure,dΓn::Measure,ptype::ProblemType)
  navier_stokes_functions(dΩ,dΓn,issteady(ptype))
end

function navier_stokes_functions(
  dΩ::Measure,
  dΓn::Measure,
  ::Val{true})

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
